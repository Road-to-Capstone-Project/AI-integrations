
# INSTALLATION
## Prerequisite
- Conda (or Miniconda)
- Open Terminal in Conda `base` environment  

## CLI on Windows
```bash
conda create --prefix "absolute-path:\to\your\app\dir\AI-integrations\.venv" python=3.10

conda activate absolute-path:\to\your\app\dir\.venv

cd absolute-path:\to\your\app\dir\AI-integrations

conda clean -a

pip cache purge

conda install _tflow_select=2.3.0 abseil-cpp=20211102.0 aiohappyeyeballs=2.4.4 aiohttp=3.11.10 aiosignal=1.2.0 astunparse=1.6.3 async-timeout=5.0.1 attrs=24.3.0 blas=1.0 blinker=1.9.0 brotli-python=1.0.9 bzip2=1.0.8 c-ares=1.19.1 ca-certificates=2025.2.25 cachetools=5.5.1 certifi=2025.4.26 cffi=1.17.1 charset-normalizer=3.3.2 click=8.1.8 colorama=0.4.6 cryptography=41.0.3 flask=3.1.0 flask-cors=3.0.10 flatbuffers=2.0.0 frozenlist=1.5.0 gast=0.4.0 giflib=5.2.2 google-auth=2.38.0 google-auth-oauthlib=0.4.4 google-pasta=0.2.0 grpc-cpp=1.48.2 grpcio=1.48.2 h5py=3.12.1 hdf5=1.14.5 icc_rt=2022.1.0 icu=58.2 idna=3.7 intel-openmp=2023.1.0 itsdangerous=2.2.0 jinja2=3.1.6 jpeg=9e keras-preprocessing=1.1.2 libcurl=8.9.1 libffi=3.4.4 libpng=1.6.39 libprotobuf=3.20.3 libssh2=1.10.0 markdown=3.8 markupsafe=3.0.2 mkl=2023.1.0 mkl-service=2.4.0 mkl_fft=1.3.11 mkl_random=1.2.8 multidict=6.1.0 numpy=1.26.4 numpy-base=1.26.4 oauthlib=3.2.2 openssl=1.1.1w opt_einsum=3.3.0 packaging=24.2 pip=25.1 propcache=0.3.1 pyasn1=0.4.8 pyasn1-modules=0.2.8 pycparser=2.21 pyjwt=2.10.1 pyopenssl=23.2.0 pysocks=1.7.1 python=3.10.13 python-flatbuffers=24.3.25 re2=2022.04.01 requests=2.32.3 requests-oauthlib=2.0.0 rsa=4.7.2 scipy=1.15.2 setuptools=78.1.1 six=1.17.0 snappy=1.2.1 sqlite=3.45.3 tbb=2021.8.0 tensorboard-data-server=0.6.1 tensorboard-plugin-wit=1.8.1 tensorflow=2.10.0 tensorflow-base=2.10.0 tensorflow-estimator=2.10.0 tk=8.6.14 typing-extensions=4.12.2 typing_extensions=4.12.2 urllib3=2.3.0 vc=14.42 vs2015_runtime=14.42.34433 werkzeug=3.1.3 wheel=0.45.1 win_inet_pton=1.1.0 xz=5.6.4 yarl=1.18.0 zlib=1.2.13

pip install tensorflow-recommenders dotenv psycopg2 sqlalchemy pandas --no-cache-dir
```