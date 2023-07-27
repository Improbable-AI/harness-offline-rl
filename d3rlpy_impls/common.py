

def wrap_sampler(sampler, baseclass):
  from weighted import AW, RW
  if sampler == "uniform":
    return baseclass
  elif sampler == "RW":
    class WrappedAlgo(RW, baseclass):
      alpha = 0.1 # Set alpha here
    return WrappedAlgo
  elif sampler == "AW":
    class WrappedAlgo(AW, baseclass):
      alpha = 0.1 # Set alpha here
    return WrappedAlgo