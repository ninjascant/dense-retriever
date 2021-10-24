from ..estimators.bert_dot import BertDot


def run_inference(model_name_or_path, dataset_dir, out_dir, id_col='doc_id'):
    estimator = BertDot(
        model_name_or_path=model_name_or_path,
        train_steps=0,
        num_epochs=0,
        batch_size=32,
        accum_steps=0,
        lr=0,
        # metric_fn=compute_metrics
    )
    estimator.predict(dataset_dir=dataset_dir, out_dir=out_dir, id_col=id_col)