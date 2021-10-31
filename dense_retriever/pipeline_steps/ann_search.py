from ..transforms.ann_index import ANNIndex


def run_search_from_scratch(
        context_embedding_dir,
        query_embedding_dir,
        out_path,
        embedding_size,
        top_n,
        load_from_sub_dirs,
        index_out_path,
        metric_type='cosine'
):
    transformer = ANNIndex(
        transformer_out_path=index_out_path,
        embedding_size=embedding_size,
        top_n=top_n,
        load_from_sub_dirs=load_from_sub_dirs,
        metric_type=metric_type
    )
    transformer.fit(context_embedding_dir)
    transformer.transform(query_embedding_dir, out_path)
