import optuna


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("char_embedding_dim", 32, 256)

    trial.suggest_int("lstm_hidden_dim", 32, 256)
    trial.suggest_int("lstm_num_layers", 1, 4)

    trial.suggest_int("ff_dim", 200, 600)
    trial.suggest_int("ff_num_layers", 1, 4)

    trial.suggest_float("dropout", 0.0, 0.8)
    trial.suggest_float("lr", 5e-3, 5e-1, log=True)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,
        config_file="./configs/fg_optuna.jsonnet",
        serialization_dir=f"./result/optuna/{trial.number}",
        metrics="best_validation_accuracy"
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="optuna_allennlp",
        direction="maximize",
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=30,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    optuna.integration.allennlp.dump_best_config(
        "./config/ner_optuna_best.jsonnet", "best_imdb_optuna.json", study)
