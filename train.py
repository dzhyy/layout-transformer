import os
import torch
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter
from script.lr_scheduler import get_cosine_schedule_with_warmup
from script.dataloader import LayoutDataset
from model import language_model
from utils import logger, option, path
from utils.draw import LogPainter
from script.misc import RenderMode
from script.criterion import MutiLoss
'''
这个数据集（任务）和之前的不同之处：
图片需要编码
文字没有字数
TODOsummary: 现在模型内和数据集和图片绘制只支持x1x2y1y2
'''

def get_result_print(batch, tgt_output, step_info, painter):
    pred = torch.argmax(tgt_output[0],dim=-1)   # TODO 支持批量绘制(文本提示只有第一个即可)
    logging.info(f'epoch_{step_info[0]}/{step_info[1]}:')
    logging.info(f"framework_name: {batch.frameworks[0]['name']}")
    logging.info(f"framework_labels: {batch.frameworks[0]['labels']}")
    logging.info(f'src: {batch.src[0].cpu().numpy().tolist()}')
    logging.info(f'decoder_output_label: {batch.trg_y[0].cpu().numpy().tolist()}')
    logging.info(f'decoder_output_pred: {pred.cpu().numpy().tolist()}')
    painter.log(batch.frameworks[0], pred.cpu().numpy().tolist(), f'epoch_{step_info[0]}_')


def main(args):
    logger.set_logger(os.path.join(args.log_root,'train.log.txt'))
    device = torch.device('cpu') if args.cpu is True else torch.device('cuda:0')
    
    dataset = LayoutDataset(args, device, use_buffer=True) # Num fo samples: 3860
    train_dataset, eval_dataset = random_split(dataset,[int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=args.batch_size,collate_fn=dataset.collate_fn)
    logging.info(f'Num fo samples:{len(dataset)},train samples:{len(train_dataset)},evaluat samples:{len(eval_dataset)}')
    logging.info(f'Device:{device}')

    args.src_vocab = dataset.layout_processor.vocab_size
    args.tgt_vocab = dataset.layout_processor.vocab_size
    model = language_model.make_model(args)
    logging.info(args)
    
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.PAD,)
    criterion = MutiLoss()


    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.n_warmup_epochs, num_training_steps=args.n_epochs)

    logging.info('Start training.')
    model = model.to(device)
    path.clear_folder(os.path.join(args.log_root, "runs"))
    writer = SummaryWriter(comment='layout', log_dir=os.path.join(args.log_root, "runs"))
    log_painter = LogPainter(args, dataset.layout_processor,mode=RenderMode.IMAGE)
    # early stop
    best_perform = float('inf')
    last_epoch = 0
    stop_count = 0

    for epoch in range(1, args.n_epochs + 1):
        logging.info(f'\nepoch_{epoch}/{args.n_epochs}:')

        # model train
        model.train()
        train_losses = 0
        for batch in tqdm(train_dataloader, desc='training'):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.bbox_trg, batch.n_tokens, batch.seq_mask)
            loss.backward()

            optimizer.step()
            train_losses += loss.item()

        # model eval
        eval_losses = 0
        model.eval()
        step = 0 
        for batch in tqdm(eval_dataloader, desc='evaluating'):
            with torch.no_grad():
                output = model(batch)
                loss = criterion(output, batch.bbox_trg, batch.n_tokens, batch.seq_mask)
                
                if step%10 == 0:
                    get_result_print(batch, output, [epoch, args.n_epochs], log_painter)
                
                eval_losses += loss.item()
                step = step+1

        train_epoch_loss = train_losses / len(train_dataloader)
        eval_epoch_loss = eval_losses / len(eval_dataloader)

        logging.info(f'Train loss: {train_epoch_loss}, Eval loss: {eval_epoch_loss}')
        writer.add_scalars('loss', {'train': train_epoch_loss}, epoch)
        writer.add_scalars('loss', {'valid': eval_epoch_loss}, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()
        
        # early stop
        if(eval_losses < best_perform):
            best_perform = eval_losses
            stop_count = 0
            # model save
            torch.save(model.state_dict(), os.path.join(args.log_root,f'model.epoch_{epoch}_p.pth'))
            path.remove_file(os.path.join(args.log_root,f'model.epoch_{epoch-1}_p.pth'))
        else:
            if (stop_count > 5):
                last_epoch = epoch
                break
            stop_count = stop_count+1

    torch.save(model.state_dict(), os.path.join(args.log_root,f'model.epoch_{last_epoch}_f.pth'))
    writer.close()



def cli_main():
    # TODO:使用此函数确定是否使用DDP进行训练
    args = option.get_trainning_args()
    main(args)


if __name__=='__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cli_main()