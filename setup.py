from setuptools import setup, find_packages

setup(
    name="sparse_frontier",
    version="0.1.0",
    description="Official implementation of the Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs",
    url="https://github.com/PiotrNawrot/sparse-frontier",
    packages=find_packages(include=['sparse_frontier', 'sparse_frontier.*']),
    entry_points={
        'vllm.general_plugins':[
            "swap_vllm_attention = sparse_frontier.modelling.models.vllm_model:swap_vllm_attention"
        ]
    },
    install_requires=[
        "transformers==4.48.0",
        "datasets==3.2.0",
        "flash-attn==2.7.3",
        "vllm==0.6.6.post1",
        "accelerate==1.3.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "matplotlib==3.10.0",
        "wonderwords",
        "gitpython",
        "pandas",
        "seaborn",
        "statsmodels",
        "pyyaml",
        "numpy",
    ],
    python_requires=">=3.10",
)
