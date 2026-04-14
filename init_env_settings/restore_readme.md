# 복구 세팅
echo "source /workspace/INIT_SETTING/restore_env.sh" >> ~/.bashrc
source ~/.bashrc

# 백업 및 복구
ssh vast_v2 "tar czf - /workspace" > workspace_backup_0402.tar.gz
