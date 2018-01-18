echo "setting conda env"
conda create --name mayEnv python=3 numpy matplotlib scipy numba
source activate mayEnv
echo "running test.py"
python test.py
