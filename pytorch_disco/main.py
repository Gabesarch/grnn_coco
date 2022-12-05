# import click #argparse is behaving weirdly
import argparse
import os
import cProfile
import logging
import ipdb 
st = ipdb.set_trace
logger = logging.Logger('catch_all')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["LC_ALL"]= 'C.UTF-8'

# @click.command()
# @click.argument("mode", required=True)
# @click.option("--exp_name","--en", default="trainer_basic", help="execute expriment name defined in config")
# @click.option("--run_name","--rn", default="1", help="run name")

parser = argparse.ArgumentParser(description='experiment names & modes')
parser.add_argument("--mode","--m", default="moc", help="experiment mode")
parser.add_argument("--exp_name","--en", default="trainer_basic", help="execute expriment name defined in config")
parser.add_argument("--run_name","--rn", default="1", help="run name")
args = parser.parse_args()


def main():
    mode = args.mode
    exp_name = args.exp_name
    run_name = args.run_name
    if run_name == "1":
        run_name = exp_name

    
    os.environ["exp_name"] = exp_name
    os.environ["run_name"] = run_name
    if mode:
        if "cs" == mode:
            mode = "CLEVR_STA"
            os.environ["MODE"] = mode
        elif "nel" == mode:
            mode = "NEL_STA"
            os.environ["MODE"] = mode
        elif "moc" == mode:
            mode = "CARLA_MOC"
            os.environ["MODE"] = mode
            from model_carla_moc import CARLA_MOC
        elif "dino_multiview" == mode:
            mode = "DINO_MULTIVIEW"
            os.environ["MODE"] = mode
            from model_dino_multiview import DINO_MULTIVIEW
        elif "gqn" == mode:
            mode = "CARLA_GQN"
            os.environ["MODE"] = mode
            from model_carla_gqn import CARLA_GQN
        elif "lescroart_moc" == mode:
            mode = "LESCROART_MOC"
            os.environ["MODE"] = mode
            from model_lescroart_moc import LESCROART_MOC
        elif "lescroart_gqn" == mode:
            mode = "LESCROART_GQN"
            os.environ["MODE"] = mode
        elif "omnidata_moc" == mode:
            mode = "OMNIDATA_MOC"
            os.environ["MODE"] = mode
            from model_omnidata_moc import OMNIDATA_MOC
        
    
    import hyperparams as hyp
    # from model_nel_sta import NEL_STA    
    # from model_clevr_sta import CLEVR_STA
    
    
    
    

    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)

    if hyp.do_clevr_sta:
        log_dir_ = os.path.join("logs_clevr_sta", hyp.name)    
    elif hyp.do_nel_sta:
        log_dir_ = os.path.join("logs_nel_sta", hyp.name)
    elif hyp.do_carla_moc:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_carla_gqn:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_dino_multiview:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_lescroart_moc:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_lescroart_gqn:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_omnidata_moc:
        log_dir_ = os.path.join("logs_omnidata_moc", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)
    # st()
    # try:
    if hyp.do_clevr_sta:
        model = CLEVR_STA(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()        
    elif hyp.do_nel_sta:
        model = NEL_STA(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    elif hyp.do_carla_moc:
        model = CARLA_MOC(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    elif hyp.do_carla_gqn:
        model = CARLA_GQN(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    # elif hyp.do_lescroart_moc:
    #     model = LESCROART_MOC(checkpoint_dir=checkpoint_dir_,
    #             log_dir=log_dir_)
    #     model.go()
    elif hyp.do_lescroart_gqn:
        model = CARLA_GQN(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    elif hyp.do_dino_multiview:
        model = DINO_MULTIVIEW(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    elif hyp.do_omnidata_moc:
        model = OMNIDATA_MOC(checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
        model.go()
    else:
        assert(False) # what mode is this?

    # except (Exception, KeyboardInterrupt) as ex:
    #     logger.error(ex, exc_info=True)
    #     st()
    #     log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

