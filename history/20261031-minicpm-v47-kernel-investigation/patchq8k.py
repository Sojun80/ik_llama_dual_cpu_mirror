import pathlib,re,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text()
for name in ['Q4_K','Q5_K','Q6_K']:
 pat=r'(\[GGML_TYPE_'+name+r'\] = \{)(.*?)(\n    \},)';m=re.search(pat,s,re.S);assert m
 b=m.group(2);rx=r'#ifdef __AVX2__\s*\.vec_dot_type\s*=\s*GGML_TYPE_Q8_2_X4,\s*#else\s*\.vec_dot_type\s*=\s*GGML_TYPE_Q8_[01]_X4,\s*#endif';b,n=re.subn(rx,'.vec_dot_type             = GGML_TYPE_Q8_K,',b);assert n==1,(name,n);s=s[:m.start(2)]+b+s[m.end(2):]
p.write_text(s)
