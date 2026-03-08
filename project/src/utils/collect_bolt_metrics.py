import apple_bolt as bolt

bolt_id = "4rkpchk6wq"

# Get task object
task = bolt.get_task(bolt_id)

# Get all metrics
metrics = task.get_metrics()[bolt_id]


pass