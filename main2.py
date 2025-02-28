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
    parser.add_argument('--epochs', type=int, default=1,  
                        help='Number of training epochs')
    parser.add_argument('--resume', type=lambda x: bool(strtobool(x)), default=False,  
                        help='Resume training from the last checkpoint')
    parser.add_argument('--test_output_dir', type=str, 
                        default='./result/', 
                        help='Directory for test outputs')
    # Diretório para checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./logs/lightning_logs/version_0/checkpoints/', 
                        help='Directory for saving checkpoints')

    # Diffusion Process Configuration
    parser.add_argument('--time_step', type=int, default=1000, 
                        help='Number of diffusion time steps')

    # Betas Configuration
    parser.add_argument('--linear_start', type=float, default=1e-6, 
                        help='Starting value for linear beta schedule')
    parser.add_argument('--linear_end', type=float, default=1e-2, 
                        help='Ending value for linear beta schedule')

    # UNet Configuration
    parser.add_argument('--channel_in', type=int, default=7, 
                        help='Input channels for UNet')
    parser.add_argument('--channel_out', type=int, default=3, 
                        help='Output channels for UNet')
    parser.add_argument('--channel_mult', nargs='+', type=int, 
                        default=[1, 2, 4, 8], 
                        help='Channel multipliers for UNet')
    parser.add_argument('--attention_head', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--cbam', type=bool, default=False, 
                        help='Enable or disable CBAM')

    # Optimizer Configuration
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, 
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=1, 
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay value')

    # Data Paths
    parser.add_argument('--train_reference_path', type=str, 
                        default='/data/Anime/train_data/reference/', 
                        help='Path to reference data')
    parser.add_argument('--train_condition_path', type=str, 
                        default='/data/Anime/train_data/sketch/', 
                        help='Path to condition data')
    parser.add_argument('--test_reference_path', type=str, 
                        default='/data/Anime/test_data_shuffled/reference/', 
                        help='Path to reference data')
    parser.add_argument('--test_condition_path', type=str, 
                        default='/data/Anime/test_data_shuffled/sketch/', 
                        help='Path to condition data')

    # Batch Sizes
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                        help='Batch size for validation')

    # Image Size
    parser.add_argument('--size', type=int, default=256, 
                        help='Image size')

    parser.add_argument('--gpus', nargs='+', type=int, default=[1])

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    cfg = parse_arguments()

    # Fixar aleatoriedade para garantir resultados determinísticos
    pl.seed_everything(42, workers=True)

    # Garantir que a pasta de checkpoints existe
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Model checkpoint callback (salva apenas o último)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        monitor="train_avg_loss",
        filename='{epoch:02d}-{train_avg_loss:.4f}',
        save_top_k=1,  
        mode="min",
        every_n_epochs=1,
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger("logs")

    # Trainer configuration com precisão determinística
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
        deterministic=True,
    )

    # Instanciando o modelo
    model = AnimeDiffusion(cfg)  # <-- ✅ Agora 'linear_start' e outros parâmetros estão no cfg!

    # Carregar checkpoint se --resume for True
    import os

    # Defina um caminho inicial, caso você queira começar a busca de uma pasta específica
    start_folder = '/kaggle/working/logs/'  # Defina o caminho inicial da pasta de busca

    # Carregar checkpoint se --resume for True
    last_checkpoint = None
    if cfg.resume and os.path.exists(cfg.checkpoint_dir):
        # Modifique aqui: busque apenas dentro da pasta start_folder
        checkpoints = sorted(
            [f for f in os.listdir(start_folder) if f.endswith(".ckpt")],
            key=lambda x: os.path.getmtime(os.path.join(start_folder, x))
        )
        
        if checkpoints:
            last_checkpoint = os.path.join(start_folder, checkpoints[-1])  
            print(f"✅ Retomando do checkpoint: {last_checkpoint}")
    

    # Treinamento
    if cfg.do_train:
        trainer.fit(model, ckpt_path=last_checkpoint if last_checkpoint else None)

    # Teste
    if cfg.do_test:
        os.makedirs(cfg.test_output_dir, exist_ok=True)
        trainer.test(model, ckpt_path=last_checkpoint if last_checkpoint else None)
