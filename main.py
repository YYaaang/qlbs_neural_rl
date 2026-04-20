# ============================================================
# main.py
# ============================================================
from configs.base.env_cfg import device, torch_dtype
from init_train_models import train_new_models, clear
from add_new_critic_models import train_new_critic_from_existing_actor_and_critic0
def main():
    # -------------------------------------------------
    # 1, New critic0, critic, actor models
    # -------------------------------------------------
    OPTION_MODEL_TYPE = 'bs'
    from configs.full_bs_config import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()

    OPTION_MODEL_TYPE = 'heston'
    from configs.analysis_heston.full_heston_config_1 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()

    # -------------------------------------------------
    # 2, Add critic models
    # -------------------------------------------------
    all_new_L = [0.1, 0.5, 1, 10, 20, 50, 100, 1000]

    OPTION_MODEL_TYPE = 'bs'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/bs_0_15_",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()

    # add critic
    OPTION_MODEL_TYPE = 'heston'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/heston_V00_04__kappa1_5__theta0_04__sigma0_25__rho_0_5__trans_cost0_000_1",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()

    # -------------------------------------------------
    # 1, New critic0, critic, actor models
    # -------------------------------------------------
    OPTION_MODEL_TYPE = 'bs'
    from configs.analysis_bs.full_bs_config_0_2 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()
    OPTION_MODEL_TYPE = 'bs'
    from configs.analysis_bs.full_bs_config_0_3 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()
    OPTION_MODEL_TYPE = 'bs'
    from configs.analysis_bs.full_bs_config_0_4 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()
    ############################################
    # OPTION_MODEL_TYPE = 'heston'
    # from configs.full_heston_config_2 import full_cfg as full_cfg
    # train_new_models(full_cfg, OPTION_MODEL_TYPE)
    # clear()
    OPTION_MODEL_TYPE = 'heston'
    from configs.analysis_heston.full_heston_config_3 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()
    OPTION_MODEL_TYPE = 'heston'
    from configs.analysis_heston.full_heston_config_4 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()
    OPTION_MODEL_TYPE = 'heston'
    from configs.analysis_heston.full_heston_config_5 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()

    ############################################
    OPTION_MODEL_TYPE = 'bs'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/5_1_bs_0_2_mu_0_08",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()

    ############################################
    OPTION_MODEL_TYPE = 'heston'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/5_1_bs_0_2_mu_0_08",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()

    # -------------------------------------------------
    # 1, New critic0, critic, actor models
    # -------------------------------------------------
    OPTION_MODEL_TYPE = 'bs'
    from configs.analysis_bs.full_bs_config_0_2_mu_0_3 import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)
    clear()

    # -------------------------------------------------
    # 2, Add critic models
    # -------------------------------------------------
    all_new_L = [0.1, 0.5, 1, 10, 20, 50, 100, 1000]

    OPTION_MODEL_TYPE = 'bs'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/4_2_bs_0_2",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()

if __name__ == "__main__":


    main()
