from model import build_transformer




build_transformer(src_vocab_size=1000,tgt_vocab_size=1000,src_seq_len=20,tgt_seq_len =20,d_model=512,N=6,h=8,dropout=0.1,d_ff=2048)
