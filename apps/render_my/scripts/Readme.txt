
 
先在脚本中设置ASSETS_ROOT路径， 然后在conda 环境中安装requirements.txt中的库；



source  activate  /data/conda_env/infinigen/
python download_assets_for_OptiX.py   hdri --fmt  hdr   --flatten     --all
python download_assets_for_OptiX.py   pbr  --all
python download_assets_for_OptiX.py objaverse \
  --opp --limit 36500 \
  --proc 2 --batch-size 100 \
  --cache-dir /data/codes/optix_all/cache_tmp \
  --link-mode symlink
