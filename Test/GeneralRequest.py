try:
r = request.get(url,timeout=30)
r.raise_for_status()
r.encoding = r.apparent_encoding
return r.text
exception:
    return "Exception"