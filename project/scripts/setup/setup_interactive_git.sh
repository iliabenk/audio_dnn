# update git email and name for interactive sessions
IS_INTERACTIVE=$(python -c "import turibolt as bolt; print(bolt.get_task(bolt.get_current_task_id()).is_interactive)")

if [ "$IS_INTERACTIVE" == "True" ]; then
  echo "Interactive task, disabling iris profiler.."
  echo 'export IRIS_PROFILER=0' >> ~/.bashrc

  echo "Interactive task, configuring git..."

  # Get the owner username
  OD_USERNAME=$(python -c "import turibolt as bolt; print(bolt.get_task(bolt.get_current_task_id()).owners)")

  # Check if OD_USERNAME is "ibenkovitch" and adjust accordingly
  if [ "$OD_USERNAME" == "ibenkovitch" ]; then
    git config --global user.name "Ilia Benkovitch"
  fi

  # Set the email using the adjusted OD_USERNAME
  git config --global user.email "$OD_USERNAME@apple.com"

  git config --global --add safe.directory /mnt/task_runtime
else
  echo "Non-interactive task, ignoring git configuration and installing relevant packages"
fi