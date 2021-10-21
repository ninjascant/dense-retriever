from ..transforms.preprocessing import TrainSetConstructor, TrainSetTokenizer, TestSetTokenizer, TextExtractor


def tokenize_train_set(train_set_path, tokenizer_name_or_path, use_cache, out_path):
    transformer = TrainSetTokenizer(tokenizer_name_or_path, max_length=512, padding='max_length', use_cache=use_cache)
    transformer.transform(train_set_path, out_path)


def tokenize_test_set(test_set_path, tokenizer_name_or_path, out_path, text_col_name, max_length):
    transformer = TestSetTokenizer(tokenizer_name_or_path, max_length=max_length, padding='max_length',
                                   text_column=text_col_name)
    transformer.transform(test_set_path, out_path)


def extract_texts_for_inference(input_path, out_path, id_col_name, text_col_name):
    transformer = TextExtractor(text_col_name=text_col_name, id_col_name=id_col_name)
    transformer.transform(input_path, out_path)


def construct_train_set(search_result_file, query_sample_file, train_docs_file, out_path):
    transformer = TrainSetConstructor(query_sample_file=query_sample_file, train_docs_file=train_docs_file)
    transformer.transform(search_result_file, out_path)








