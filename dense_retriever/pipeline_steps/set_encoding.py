from ..transforms.set_encoding import TripletTokenizer


def tokenize_train_triplets(tokenizer_name_or_path: str, triplet_file_path: str, out_path: str):
    transformer = TripletTokenizer(tokenizer_name_or_path=tokenizer_name_or_path)
    transformer.transform(triplet_file_path, out_path)