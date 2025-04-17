export PATH=/root/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"

grep -qxF 'export DISABLE_AUTO_UPDATE=true' ~/.zshrc || sed -i '1s;^;export DISABLE_AUTO_UPDATE=true\n;' ~/.zshrc

conda activate verl
pip install torch-memory-saver unidiff -i https://mirrors.ustc.edu.cn/pypi/simple
pip install liger_kernel -i https://mirrors.ustc.edu.cn/pypi/simple
# sglang backend
# pip install "sglang[all]>=0.4.5" -i https://mirrors.ustc.edu.cn/pypi/simple
# vllm backend
pip install vllm==0.8.2 -i https://mirrors.ustc.edu.cn/pypi/simple
pip install tensordict==0.6.2 -i https://mirrors.ustc.edu.cn/pypi/simple

# Check if this node is the master
if [ "$(hostname)" = "${MASTER_ADDR}" ]; then

ray start --head --node-ip-address ${MASTER_ADDR} --port ${MASTER_PORT}

# Can put any rl scripts here to directly submit jobs

else

ray start --address ${MASTER_ADDR}:${MASTER_PORT} --block

fi

# avoid low gpu usage, comment out if submitting jobs in script
python /mnt/data/fuxiang/llm-verifier/evaluation-kit/gpu_idle.py --threshold 0.0
