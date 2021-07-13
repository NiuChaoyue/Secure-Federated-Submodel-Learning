# Secure Federated Submodel Learning

<a name="bDN7d"></a>
# 1. General introduction
In this project, we build a simulation platform in Python, for testing our work on Secure Federated Submodel Learning (SFSL) which was accepted by ACM MobiCom 2020, and for comparing with Google's Federated Learning (FL)-based baselines. Two key advantages of our code framework over many existing FL codes lie in (1) the socket communication module between the cloud server and each client; and (2) the security and privacy preservation modules, such as secret sharing-based secure aggregation. In particular, these two modules are of significant importance to go from simulation to deployment, according to our deployment practice on Android, iOS, Linux, and embedded devices with Alibaba's [MNN](https://github.com/alibaba/MNN) for on-device training. Further, as an exemplary application, we use the click-through-rate (CTR) prediction task, the Deep Interest Network (DIN) model, and Taobao dataset. If you find our work and code useful, please consider citing our papers as follows:
<br />[MobiCom Version] Chaoyue Niu, Fan Wu, Shaojie Tang, Lifeng Hua, Rongfei Jia, Chengfei Lv, Zhihua Wu, and Guihai Chen, Billion-Scale Federated Learning on Mobile Clients: A Submodel Design with Tunable Privacy, in Proc. of MobiCom, pp. 405 - 418, 2020. [[PDF](https://dl.acm.org/doi/10.1145/3372224.3419188), [Slides](https://niuchaoyue.github.io/res/ppt/MOBICOM20.pptx), [Video](https://www.youtube.com/watch?v=V1Wgqvcy-Pk&ab_channel=ACMSIGMOBILEONLINE)]
<br />[ArXiv Version] Chaoyue Niu, Fan Wu, Shaojie Tang, Lifeng Hua, Rongfei Jia, Chengfei Lv, Zhihua Wu, and Guihai Chen, Secure Federated Submodel Learning, arXiv: 1911.02254. [[PDF](http://arxiv.org/abs/1911.02254)]
<a name="EG7kL"></a>
# 2. File instruction
**taobao_data_process/**

- process raw Taobao log for centralized learning
- split training set by user for Federated Submodel Learning (FSL)

**Notes:** Our 30-day Taobao dataset cannot be released due to the restriction of Alibaba. Yet, some public datasets for CTR prediction are available from [Alimama](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56), [Amazon](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/), etc. <br />**plain/**

- Centralized_DIN: train DIN with SGD
- FedAvg (Federated Averaging): averaging the chosen clients' full model updates with the weights being their full training set sizes  
- FedSubAvg (Federated Submodel Averaging): averaging the chosen clients' submodel updates with the weights being their relevant training set sizes

**Notes:** The implementation of DIN mainly refers to the released code of Deep Interest Evolution Network ([DIEN](https://github.com/mouna99/dien)).<br />**secure/**

- SFL (Secure FL): FL with secure aggregation
- SFSL: FSL with tunable local differential privacy

**Notes:** Secure aggregation is based on secret sharing. The building blocks of our SFSL include Private Set Union (PSU) (that builds on Bloom filter, randomization, and secure aggregation), randomized response, and secure aggregation. 
<a name="AkcYv"></a>
# 3. Dependence
We suggest you to manage Python's enviroments and packages mainly with Anaconda and with its pip just as an alternative (i.e., if the package is not available from anaconda). The full list of packages that we use is shown as follows:<br />(fed) root@niuchaoyue:~# conda list<br /># packages in environment at /root/anaconda2/envs/fed:<br />#<br /># Name                          Version                   Build  Channel<br />_libgcc_mutex                 0.1                          main<br />_tflow_select                   2.1.0                       gpu<br />absl-py                           0.8.0                        py27_0<br />asn1crypto                     0.24.0                      py27_0<br />astor                               0.8.0                        py27_0<br />backports                       1.0                           py_2<br />backports.weakref          1.0.post1                 py_1<br />blas                                 1.0                          mkl<br />c-ares                              1.15.0                     h7b6447c_1001<br />ca-certificates                  2019.8.28               0<br />**certifi**                             2019.9.11               py27_0<br />cffi                                   1.12.3                     py27h2e261b9_0<br />**chardet**                           3.0.4                       py27_1003<br />click                                 7.0                          py27_0<br />**cryptography**                 **2.7**                         py27h1ba5d50_0<br />cudatoolkit                      9.2                          0<br />cudnn                              7.6.0                       cuda9.2_0<br />cupti                                9.2.148                   0<br />**dnspython**                      1.16.0                     py27_0<br />enum34                           1.1.6                       py27_1<br />**eventlet**                           0.25.1                    pypi_0    pypi     <pip install><br />**flask**                                 1.1.1                      py_0<br />**flask-socketio**                 4.2.1                      py_0<br />funcsigs                            1.0.2                      py27_0<br />futures                              3.3.0                      py27_0<br />gast                                  0.3.2                      py_0<br />gmp                                  6.1.2                     h6c8ec71_1<br />**gmpy2**                             2.0.8                     py27h10f8cd9_2<br />**greenlet**                           0.4.15                    py27h7b6447c_0<br />grpcio                               1.16.1                    py27hf8bcb03_1<br />h5py                                  2.9.0                     py27h7918eee_0<br />hdf5                                  1.10.4                    hb1b8bf9_0<br />**idna**                                  2.8                         py27_0<br />intel-openmp                    2019.4                   243<br />ipaddress                          1.0.22                     py27_0<br />**itsdangerous**                   1.1.0                       py27_0<br />**jinja2**                                2.10.1                     py27_0<br />**keras-applications**          **1.0.8**                       py_0<br />**keras-preprocessing**       **1.0.9**                       py_0<br />libedit                               3.1.20181209          hc058e9b_0<br />libffi                                  3.2.1                        hd88cf55_4<br />libgcc-ng                           9.1.0                       hdf63c60_0<br />libgfortran-ng                   7.3.0                        hdf63c60_0<br />libprotobuf                        3.9.2                       hd408876_0<br />libstdcxx-ng                       9.1.0                       hdf63c60_0<br />linecache2                         1.0.0                       py27_0<br />markdown                         3.1.1                       py27_0<br />**markupsafe**                      1.1.1                       py27h7b6447c_0<br />mkl                                    2019.4                    243<br />mkl-service                        2.3.0                       py27he904b0f_0<br />mkl_fft                               1.0.14                     py27ha843d7b_0<br />mkl_random                      1.1.0                       py27hd6b4f25_0<br />mock                                 3.0.5                       py27_0<br />**monotonic**                       1.5                          py_0<br />mpc                                   1.1.0                       h10f8cd9_1<br />mpfr                                  4.0.1                       hdf1c602_3<br />ncurses                              6.1                          he6710b0_1<br />numpy                               1.16.4                     py27h7e9f1db_0<br />numpy-base                      1.16.4                     py27hde5b4d6_0<br />openssl                              1.1.1d                     h7b6447c_2<br />**pandas**                              **0.24.2**                    py27he6710b0_0<br />pip                                     19.2.3                     py27_0<br />protobuf                            3.9.2                       py27he6710b0_0<br />pycparser                           2.19                       py27_0<br />**pycryptodome**                 **3.8.2**                      py27hb69a4c5_0<br />pyopenssl                          19.0.0                     py27_0<br />pysocks                              1.7.1                      py27_0<br />**python**                              **2.7.16**                    h9bab390_7<br />python-dateutil                  2.8.0                      py27_0<br />**python-engineio**              3.9.3                      py_0<br />**python-socketio**               4.3.1                      py_0<br />pytz                                    2019.2                   py_0<br />readline                              7.0                         h7b6447c_5<br />**requests                            **2.22.0                     py27_0<br />scipy                                  1.2.1                       py27h7c811a0_0<br />**secretsharing**                    **0.2.6**                     pypi_0    pypi    <pip install><br />setuptools                          41.2.0                    py27_0<br />**six**                                      1.12.0                    py27_0<br />sqlite                                  3.29.0                    h7b6447c_0<br />tensorboard                       1.12.2                    py27he6710b0_0<br />tensorflow                          1.12.0                    gpu_py27h2a0f108_0<br />tensorflow-base                 1.12.0                    gpu_py27had579c0_0<br />**tensorflow-gpu**                **1.12.0**                   h0d30ee6_0<br />termcolor                           1.1.0                      py27_1<br />tk                                        8.6.8                      hbc83047_0<br />traceback2                          1.4.0                      py27_0<br />unittest2                             1.1.0                      py27_0<br />**urllib3**                                1.24.2                    py27_0<br />**utilitybelt**                           0.2.6                     pypi_0    pypi    <pip install><br />**werkzeug**                          0.16.0                    py_0<br />wheel                                  0.33.6                    py27_0<br />zlib                                      1.2.11                    h7b6447c_3
<a name="v6CGE"></a>
# 4. Acknowledgement
We would like to sincerely thank Renjie Gu, Hongtao Lv, and Hejun Xiao, mainly for their contributions to the data processing, the socket communication, and the model compression modules. We also want to thank the edge-AI group in Taobao for great engineering support.
