"""Passive event-driven Colab occupancy; no threads, timers or keepalive."""

from __future__ import annotations

from pathlib import Path

_OBSERVER = """def _klab_install_kernel_observer(payload):
 import json, os, pathlib, time
 p=json.loads(payload); root=pathlib.Path(p["worker_root"])
 if not root.is_absolute() or root.is_symlink(): raise RuntimeError("invalid observer root")
 root.mkdir(parents=True,exist_ok=True,mode=0o700)
 if root.stat().st_mode & 0o077: raise RuntimeError("observer root must be private")
 status=root/"kernel-status.json"
 boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
 def start_time():
  data=pathlib.Path("/proc/self/stat").read_text(); fields=data[data.rfind(")")+2:].split()
  return int(fields[19])
 def publish(state):
  value={"kernel_status_version":1,"instance_id":p["instance_id"],"boot_id":boot,"pid":os.getpid(),"process_start":start_time(),"state":state,"observed_at":time.time()}
  temporary=status.with_name(status.name+".tmp-"+str(os.getpid()))
  descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  try:
   with os.fdopen(descriptor,"w",encoding="utf-8") as output:
    json.dump(value,output,sort_keys=True,separators=(",",":"));output.flush();os.fsync(output.fileno())
   os.replace(temporary,status)
  finally: temporary.unlink(missing_ok=True)
 try: shell=get_ipython()
 except NameError: raise RuntimeError("kernel observer requires IPython")
 key=p["instance_id"]+":"+str(root)
 hooks=getattr(shell,"_klab_kernel_observers",{})
 if key not in hooks:
  def before(info): publish("BUSY")
  def after(result): publish("IDLE")
  shell.events.register("pre_run_cell",before);shell.events.register("post_run_cell",after)
  hooks[key]=(before,after);shell._klab_kernel_observers=hooks
 publish("BUSY")
"""


def notebook_cell_program(root: Path) -> str:
    """Produce a user-run enrollment cell; never submit unknown kernel work."""
    from klab.hosted.bootstrap import _program

    if not root.is_absolute():
        raise ValueError("notebook enrollment root must be absolute")
    return (
        _program("enroll", {"root": str(root)}, 60)
        + _OBSERVER
        + "_klab_install_kernel_observer(json.dumps({'worker_root': _klab_result['worker_root'], 'instance_id': _klab_result['instance_id']}))\n"
        + "del _klab_install_kernel_observer\n"
    )
