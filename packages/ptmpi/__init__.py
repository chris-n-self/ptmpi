def _run_here(cmd):
  import subprocess
  import os
  cwd = os.path.dirname(os.path.abspath(__file__))
  return subprocess.check_output(cmd, cwd=cwd)

def get_git_revision_hash():
  return _run_here(['git', 'rev-parse', 'HEAD']).strip()

def get_git_revision_short_hash():
  return _run_here(['git', 'rev-parse', '--short', 'HEAD']).strip()

def get_git_describe():
  return _run_here(['git', 'describe', '--always']).strip()

def get_version_string():
  return get_git_describe()

if __name__ == '__main__':
  print 'Hash:      "{}"'.format(get_git_revision_hash())
  print 'ShortHash: "{}"'.format(get_git_revision_short_hash())
  print 'Desc:      "{}"'.format(get_git_describe())
