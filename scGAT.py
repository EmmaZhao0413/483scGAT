import time
import os
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
from model import VAE
from util_function import *
from graph_function import *
from benchmark_util import *
import torch.multiprocessing as mp
from gae.model import GCNModelVAE, GATModelVAE, MultiHeadGAT,GCNModelAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Main entrance of scGAT')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='For 10X: folder name of 10X dataset; For CSV: csv file name')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='Directory of dataset: default(/home/wangjue/biodata/scData/10x/6/)')

parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 12800)')
parser.add_argument('--Regu-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train in Feature Autoencoder initially (default: 500)')
parser.add_argument('--EM-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train Feature Autoencoder in iteration EM (default: 200)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--quickmode', action='store_true', default=False,
                    help='whether use quickmode, skip Cluster Autoencoder (default: no quickmode)')
parser.add_argument('--cluster-epochs', type=int, default=200, metavar='N',
                    help='number of epochs in Cluster Autoencoder training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable GPU training. If you only have CPU, add --no-cuda in the command line')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='noregu',
                    help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--reduction', type=str, default='sum',
                    help='reduction type: mean/sum, default(sum)')
parser.add_argument('--model', type=str, default='VAE',
                    help='VAE/AE (default: AE)')
parser.add_argument('--gammaPara', type=float, default=0.1,
                    help='regulized intensity (default: 0.1)')
parser.add_argument('--alphaRegularizePara', type=float, default=0.9,
                    help='regulized parameter (default: 0.9)')

# Build cell graph
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStatsSingleThread)')

# Debug related
parser.add_argument('--precisionModel', type=str, default='Float',
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--coresUsage', type=str, default='1',
                    help='how many cores used: all/1/... (default:1)')
parser.add_argument('--outputDir', type=str, default='outputDir/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--saveinternal', action='store_true', default=False,
                    help='whether save internal interation results or not')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
parser.add_argument('--nonsparseMode', action='store_true', default=False,
                    help='SparseMode for running for huge dataset')

# LTMG related
parser.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                    help='expression File after ltmg in csv')

# Clustering related
parser.add_argument('--useGAEembedding', action='store_true', default=True,
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False,
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--n-clusters', default=20, type=int,
                    help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)')
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')

# imputation related
parser.add_argument('--EMregulized-type', type=str, default='Celltype',
                    help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR/Celltype')
parser.add_argument('--gammaImputePara', type=float, default=0.0,
                    help='regulized parameter (default: 0.0)')
parser.add_argument('--graphImputePara', type=float, default=0.3,
                    help='graph parameter (default: 0.3)')
parser.add_argument('--celltypeImputePara', type=float, default=0.1,
                    help='celltype parameter (default: 0.1)')
parser.add_argument('--L1Para', type=float, default=1.0,
                    help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0,
                    help='L2 regulized parameter (default: 0.001)')
parser.add_argument('--EMreguTag', action='store_true', default=False,
                    help='whether regu in EM process')
parser.add_argument('--sparseImputation', type=str, default='nonsparse',
                    help='whether use sparse in imputation: sparse/nonsparse (default: nonsparse)')

# dealing with zeros in imputation results
parser.add_argument('--zerofillFlag', action='store_true', default=False,
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--noPostprocessingTag', action='store_false', default=True,
                    help='whether postprocess imputated results, default: (True)')
parser.add_argument('--postThreshold', type=float, default=0.01,
                    help='Threshold to force expression as 0, default:(0.01)')

# GAE related
parser.add_argument('--GAEmodel', type=str,
                    default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001,
                    help='Initial learning rate for regularization.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.sparseMode = not args.nonsparseMode

# TODO
# As we have lots of parameters, should check args
checkargs(args)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device:'+str(device))

if not args.coresUsage == 'all':
    torch.set_num_threads(int(args.coresUsage))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# print(args)
start_time = time.time()

# load scRNA in csv
print('---0:00:00---scRNA starts loading.')
data, genelist, celllist = loadscExpression(
    args.datasetDir+args.datasetName+'/'+args.ltmgExpressionFile, sparseMode=args.sparseMode)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---scRNA has been successfully loaded')

scData = scDataset(data)
train_loader = DataLoader(
    scData, batch_size=args.batch_size, shuffle=False, **kwargs)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---TrainLoader has been successfully prepared.')



# Original
model = VAE(dim=scData.features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---Pytorch model ready.')
def train(epoch, train_loader=train_loader, EMFlag=False, taskType='celltype', sparseImputation='nonsparse'):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
        taskType: celltype or imputation
    '''
    model.train()
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        regulationMatrixBatch = None

        
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                    regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
    
        # L1 and L2 regularization
        # 0.0 for no regularization
        l1 = 0.0
        l2 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
        loss = loss + args.L1Para * l1 + args.L2Para * l2

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        # for batch
        if batch_idx == 0:
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return recon_batch_all, data_all, z_all



def GAEembedding(z, adj, args):
    '''
    GAE embedding for clustering
    Param:
        z,adj
    Return:
        Embedding from graph
    '''   
    # featrues from z
    # Louvain
    features = z
    # features = torch.DoubleTensor(features)
    features = torch.FloatTensor(features)

    # Old implementation
    # adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.DoubleTensor(adj_label.toarray())
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.GAEhidden1, args.GAEhidden2, 0)
    optimizer = optim.Adam(model.parameters(), lr=args.GAElr)

    hidden_emb = None
    loss_list = []
    roc_list = []
    for epoch in tqdm(range(args.GAEepochs)):
        t = time.time()
        # mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption before training: '+str(mem))
        model.train()
        optimizer.zero_grad()
        z, mu, logvar = model(features, adj_norm)


        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        loss_list.append(cur_loss)
        hidden_emb = mu.data.numpy()
        # TODO, this is prediction 
        # print(hidden_emb)
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        tqdm.write('ROC score: ' + str(roc_curr))
        roc_list.append(roc_curr)
        ap_curr = 0

        tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
            epoch + 1, cur_loss,
            ap_curr, time.time() - t))


    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    plt.plot(np.array(list(range(1,201))), np.array(roc_list))
    plt.xlabel("Epochs")
    plt.ylabel("ROC Score")
    plt.title("Training ROC Score over Epochs")
    plt.savefig("train_roc_VAEGCNModelVAE0.005.png") 
    save_df = pd.DataFrame([list(range(1,201)), roc_list])
    save_df.to_csv("train_roc_VAEGCNModelVAE0.005.csv")
    return hidden_emb


if __name__ == "__main__":
    start_time = time.time()
    adjsample = None
    celltypesample = None
    # If not exist, then create the outputDir
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    # store parameter
    stateStart = {
        # 'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    print('Start training...')
    for epoch in range(1, 20):
        recon, original, z = train(epoch, EMFlag=False)

    zOut = z.detach().cpu().numpy()
    print('zOut ready at ' + str(time.time()-start_time))
    ptstatus = model.state_dict()

    # Store reconOri for imputation
    reconOri = recon.clone()
    reconOri = reconOri.detach().cpu().numpy()

    # Step 1. Inferring celltype
    adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(
        args.k), adjTag=True)

    zDiscret = zOut > np.mean(zOut, axis=0)
    zDiscret = 1.0*zDiscret
    
    zOut = GAEembedding(zDiscret, adj, args)
    print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                    start_time)))+"---GAE embedding finished")


    resolution = 0.5
    listResult, size = generateLouvainCluster(edgeList)

    # Output final results
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---All iterations finished, start output results.')

    emblist = []
    for i in range(zOut.shape[1]):
        emblist.append('embedding'+str(i))
    embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
    embedding_df.to_csv(args.outputDir+args.datasetName+'_embedding.csv')
    graph_df = pd.DataFrame(edgeList, columns=["NodeA", "NodeB", "Weights"])
    graph_df.to_csv(args.outputDir+args.datasetName+'_graph.csv', index=False)
    results_df = pd.DataFrame(listResult, index=celllist, columns=["Celltype"])
    results_df.to_csv(args.outputDir+args.datasetName+'_results.txt')

    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+"---scGNN finished")
