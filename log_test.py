from torch.utils.tensorboard import SummaryWriter
import time

def logger_info(ret,echo):
    #tensorboard --logdir='/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/pytorch_logger --host=127.0.01

    writer = SummaryWriter('/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/pytorch_logger')
    # writer.add_scalar('return_epoch',ret,echo)

    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)
    writer.close()

    # writer.flush()

if __name__ == '__main__':
    writer = SummaryWriter('/home/wang/Desktop/RL_HarshTerrain_planning/spinningup-master/pytorch_logger',
                           flush_secs=1)
    step=0
    while(1):
        step=step+1

        x = range(1000)
        for i in x:
            time.sleep(0.1)
            writer.add_scalar('y__2x', i * 2, i)
            writer.add_scalar('y__2xx', i * 3, i)

        # writer.close()
        print('p')
