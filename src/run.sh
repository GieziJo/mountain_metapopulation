# echo "setting conda env"
# conda create --name envToRunCode python=3 numpy matplotlib scipy numba pillow
# source activate envToRunCode
echo "running metapop model"
python metapopulationSimulation.py
