import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from AnimeDiffusion import AnimeDiffusion
from distutils.util import strtobool

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration Parser')

    # Training Configuration
    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), default=True, 
                        help='Enable or disable training (True/False)')
    parser.add_argument('--do_finetuning', type=lambda x: bool(strtobool(x)), default=False, 
                        help='Enable or disable finetuning (True/False)')
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), default=False, 
                        help='Enable or disable test (True/False)')
    parser.add_argument('--epochs', type=int, default=1,  # Treina uma época por vez
                        help='Number of training epochs')
    parser.add_argument('--resume', type=lambda x: bool(strtobool(x)), default=False,  
                        help='Resume training from the last checkpoint')

    # Diretório para checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./logs/lightning_logs/version_0/checkpoints/', 
                        help='Directory for saving checkpoints')

    # Parse arguments
    args = parser.parse_args()
    return args

def clean_old_checkpoints(checkpoint_dir, keep_last=1):
    """ Mantém apenas o checkpoint mais recente e remove os antigos. """
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")],
            key=os.path.getmtime  # Ordena pelo tempo de modificação (mais antigo primeiro)
        )
        # Remove todos os checkpoints antigos, mantendo apenas os últimos 'keep_last'
        for ckpt in checkpoints[:-keep_last]:
            os.remove(ckpt)
            print(f"🗑️ Checkpoint removido: {ckpt}")

if __name__ == "__main__":
    # Parse arguments
    cfg = parse_arguments()

    # Fixar aleatoriedade para garantir resultados determinísticos
    pl.seed_everything(42, workers=True)

    # Garantir que a pasta de checkpoints existe
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Limpar checkpoints antigos antes de criar um novo
    clean_old_checkpoints(cfg.checkpoint_dir)

    # Model checkpoint callback (salva apenas o último)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="train_avg_loss",
        filename='{epoch:02d}-{train_avg_loss:.4f}',
        save_top_k=1,  # Apenas um checkpoint é salvo
        mode="min",
        every_n_epochs=1,
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger("logs")

    # Trainer configuration com precisão determinística
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=[1],
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
        deterministic=True,  # Garante que as operações sejam determinísticas
    )

    model = AnimeDiffusion(cfg)

    # Verifica se há um checkpoint salvo e continua o treinamento
    last_checkpoint = None
    if cfg.resume and os.path.exists(cfg.checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".ckpt")],
            key=lambda x: os.path.getmtime(os.path.join(cfg.checkpoint_dir, x))
        )
        if checkpoints:
            last_checkpoint = os.path.join(cfg.checkpoint_dir, checkpoints[-1])  # Pega o último checkpoint salvo
            print(f"✅ Retomando do checkpoint: {last_checkpoint}")

    # Treinamento
    if cfg.do_train:
        trainer.fit(model, ckpt_path=last_checkpoint if last_checkpoint else None)

    # Teste
    if cfg.do_test:
        os.makedirs(cfg.test_output_dir, exist_ok=True)
        trainer.test(model, ckpt_path=last_checkpoint if last_checkpoint else None)

    # Após o treinamento, remover checkpoints antigos e manter apenas o último
    clean_old_checkpoints(cfg.checkpoint_dir)
