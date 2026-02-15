#!/bin/bash
# Run this as cremer3 (the owner of task1/)
# Grants read+write+execute access to user noren2 on task1/ and its contents

setfacl -R -m u:noren2:rwX /p/scratch/training2601/CISPA_UU_HACK2/task1/

# Also set default ACL so new files created inside inherit the permission
setfacl -R -d -m u:noren2:rwX /p/scratch/training2601/CISPA_UU_HACK2/task1/

echo "Done. noren2 now has read/write access to task1/"
