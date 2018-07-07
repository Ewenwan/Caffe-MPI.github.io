# Caffe-MPI是一款分布式集群版本，目前支持GPU集群并行计算，
  Caffe-MPI(https://github.com/Caffe-MPI/Caffe-MPI.github.io)
  是一款高性能高可扩展的深度学习计算框架,是由浪潮的HPC应用开发团队进行开发。
  Caffe-MPI是一款分布式集群版本，目前支持GPU集群并行计算，
  该版本在伯克利单机单卡GPU版本上进行开发，
  其单机单卡版本信息参见 (https://github.com/BVLC/caffe, 更多细节可以访问 http://caffe.berkeleyvision.org) 。



# Caffe-MPI主要特点及优势如下：

　　

##　(1)   基于HPC系统设计

    Caffe-MPI针对HPC系统架构技术设计，硬件系统采用Lustre存储+IB网络+GPU集群，
    基于Lustre并行存储采用多进程+多线程机制并行读取训练数据，实现较高的IO吞吐；
    采用IB网络实现高速互联网，实现参数的快速传输和模型更新；
    采用数据并行机制，利用GPU集群实现大规模训练。软件编程模型采用MPI+ Multi thread+CUDA，
    节点间采用MPI通信，节点内实现CPU多线程并行和CUDA线程并行。

　　

## (2)   高性能与高可扩展性

    Caffe-MPI可以采用多机多GPU卡同时训练，每秒可以训练2000张图片，
    较BVLC单GPU卡性能实现大幅提升，并可以部署到大规模训练平台上，
    实现对大规模数据样本的训练，如Googlenet模型,Caffe-MPI较单GPU版本性能提升16倍以上，
    并支持24+ GPUs的扩展，并行效率达到72%以上。

　　

## (3)   良好的继承性与易用性

    Caffe-MPI计算框架基于伯克利的Caffe架构进行开发，
    完全保留了原始Caffe架构的特性和最新功能，并支持最新的cuDNN 5.1，
    即纯粹的C++/CUDA架构，支持命令行、Python和MATLAB接口等多种编程方式，
    具备上手快、速度快、模块化、开放性等众多特性，为用户提供了最佳的应用体验。


<html lang="en" class=" is-copy-enabled is-u2f-enabled">

<h1><a id="user-content-caffe-mpi-for-deep-learning-------------------------------------------------------------------------" class="anchor" href="#caffe-mpi-for-deep-learning-------------------------------------------------------------------------" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Caffe-MPI for Deep Learning                                                                         </h1>

<h1><a id="user-content-introduction" class="anchor" href="#introduction" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Introduction</h1>

<p>Caffe-MPI is developed by AI & HPC Application R&D team of Inspur. It is a parallel version for multi-node GPU cluster, which is designed based on the NVIDIA/Caffe forked from the BVLC/caffe ( <a href="https://github.com/NVIDIA/caffe">https://github.com/NVIDIA/caffe</a>, more details please visit <a href="http://caffe.berkeleyvision.org">http://caffe.berkeleyvision.org</a>).</p>

<h1><a id="user-content-features" class="anchor" href="#features" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Features</h1>

<h4><a id="user-content-1-the-design-basics" class="anchor" href="#1-the-design-basics" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>(1) The design basics</h4>

<p>The Caffe-MPI is designed for high density GPU clusters; The new version supports InfiniBand (IB) high speed network connection and shared storage system that can be equipped by distributed file system, like NFS and GlusterFS. The training dataset is read in parallel for each MPI process. The hierarchical communication mechanisms were developed to minimize the bandwidth requirements between computing nodes. </p>

<h4><a id="user-content-1-the-design-basics" class="anchor" href="#1-the-design-basics" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Updates of the Caffe-MPI 2.0</h4>

<p>Support NCCL 2.0 </p>

<p>Both inter and intra node GPU communication are managed by NCCL with GPU direct RDMA. </p>

<h4><a id="user-content-2-high-performance-and-high-scalability" class="anchor" href="#2-high-performance-and-high-scalability" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>(2) High performance and high scalability</h4>

<p>The AlexNet, GoogLeNet and ResNet model have been tested with Caffe-MPI 2.0 on a GPU cluster, which includes 4 nodes, and each of which has 4 P40 GPUs. The dataset is ImageNet. The speedup is 14.65X, 14.25X, 15.34X, for AlexNet (batchsize=1024), GoogLeNet (batchsize=128) and ResNet (batchsize=32) respectively on 4 nodes with 16 GPUs. </p>

<h3><a id="user-content-3-good-inheritance-and-easy-using" class="anchor" href="#3-good-inheritance-and-easy-using" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>(3) Good inheritance and easy-using</h3>

<p>Caffe-MPI retains all the features of the original Caffe architecture, namely the pure C++/CUDA architecture, support of the command line, Python interfaces, and various programming methods. As a result, the cluster version of the Caffe framework is user-friendly, fast, modularized and open, and gives users the optimal application experience.  </p>

<h1><a id="user-content-try-your-first-mpi-caffe" class="anchor" href="#try-your-first-mpi-caffe" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Try your first MPI Caffe</h1>

<p>This program can run 1 processes at least.</p>

<h4><a id="user-content-cifar10" class="anchor" href="#cifar10" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>cifar10</h4>

<ol>
<li>   Run data/cifar10/get_cifar10.sh to get cifar10 data.</li>
<li>   Run examples/cifar10/create_cifar10.sh to conver raw data to leveldb format.</li>
<li>   Run mpi_train_quick.sh to train the net. </li>
<li>   Example of mpi_train_quick.sh script.
mpirun -host node1,node2 -mca btl_openib_want_cuda_gdr 1 --mca io ompio -np 2 -npernode 1  ./build/tools/caffe train --solver=examples/cifar10/cifar10_quick_solver.prototxt --gpu=0,1,2,3 </li>
</ol>

<h1><a id="user-content-reference" class="anchor" href="#reference" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Reference</h1>

<ul>
<li><p>More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server</p></li>
<li><p>Deep Image: Scaling up Image Recognition</p></li>
</ul>

<h1><a id="user-content-ask-questions" class="anchor" href="#ask-questions" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Ask Questions</h1>

<ul>
<li><p>For reporting bugs, please use the caffe-mpi/issues page or send email to us.</p></li>
<li><p>Email address: <a href="mailto:wush@inspur.com">wush@inspur.com</a></p></li>
</ul>

<h1><a id="user-content-author" class="anchor" href="#author" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Author</h1>

<ul>
<li><p>Shaohua Wu.</p></li>
</ul>
</article>
  </div>

</div>

  </body>
</html>
