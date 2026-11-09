
def extractParameters(name_pattern):
  parameters = []
  idx = 0
  while True:
    pos = name_pattern.find("${", idx)
    if pos == -1: break
    end = name_pattern.find("}", pos)
    param_name = name_pattern[pos+2:end]
    parameters.append(param_name)
    idx = end
  return parameters

def applyParameters(pattern, parameters):
  value = pattern
  for param_name, param_value in parameters.items():
    value = value.replace("${" + param_name + "}", str(param_value))
  return value

def parameterListToDict(param_names, param_values):
  if len(param_names) == 1 and type(param_values) != list:
    param_values = [ param_values ]
  if len(param_values) != len(param_names):
    raise RuntimeError("Invalid parameter entry length")
  param_dict = {}
  for n in range(len(param_names)):
    param_dict[param_names[n]] = param_values[n]
  return param_dict
