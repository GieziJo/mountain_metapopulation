# echo "setting conda env"
# conda create --name envToRunCode python=3 numpy matplotlib scipy numba pillow rasterio pandas
# source activate envToRunCode
# conda install -c conda-forge rasterio
echo "running metapop model"
python metapopulationSimulation.py
