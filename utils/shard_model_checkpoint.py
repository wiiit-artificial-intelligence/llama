import os
import json
import argparse
import torch
import glob

def main(args):
    with open(os.path.join(args.input_model_path, 'params.json'), 'r') as fp:
        params = json.loads(fp.read())

    assert params['dim'] % args.num_shards == 0, "number of shards need to divide parameter dimension %d" % params['dim']

    print('loading...')
    checkpoints = [torch.load(path, map_location=torch.device('cpu')) for path in glob.glob(os.path.join(args.input_model_path, '*.pth'))]

    layer_kind = {
        'tok_embeddings': 'ParallelEmbedding',
        'output': 'ColumnParallelLinear',
        'attention.wq': 'ColumnParallelLinear',
        'attention.wk': 'ColumnParallelLinear',
        'attention.wv': 'ColumnParallelLinear',
        'attention.wo': 'RowParallelLinear',
        'feed_forward.w1': 'ColumnParallelLinear',
        'feed_forward.w2': 'RowParallelLinear',
        'feed_forward.w3': 'ColumnParallelLinear',
        'attention_norm': None,
        'ffn_norm': None,
        'norm': None,
        'rope.freqs': None,
    }

    output = [dict() for x in range(args.num_shards)]

    print('converting...')
    for key in checkpoints[0].keys():
        tensors = [m[key] for m in checkpoints]
        print(key)
        print('  in shapes=', [p.shape for p in tensors])
        for pattern, kind in layer_kind.items():
            if key.replace('.weight', '').endswith(pattern):
                print('  kind=', kind)
                if kind == 'ColumnParallelLinear':
                    with torch.no_grad():
                        merged = torch.cat(tensors, 0)
                        slice_size = merged.shape[0] // args.num_shards
                        for rank in range(args.num_shards):
                            output[rank][key] = merged[slice_size * rank: slice_size * (rank + 1), :].clone().detach()
                elif kind in ('ParallelEmbedding', 'RowParallelLinear'):
                    with torch.no_grad():
                        merged = torch.cat(tensors, 1)
                        slice_size = merged.shape[1] // args.num_shards
                        for rank in range(args.num_shards):
                            output[rank][key] = merged[:, slice_size * rank: slice_size * (rank + 1)].clone().detach()
                else:
                    for rank in range(args.num_shards):
                        output[rank][key] = tensors[0]
                print('  out shapes=', [output[rank][key].shape for rank in range(args.num_shards)])
                print()
                break
        else:
            raise Exception('parameter name not recognized')

    print('saving...')
    os.makedirs(args.output_model_path, exist_ok=True)
    with open(os.path.join(args.output_model_path, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params))

    for rank in range(args.num_shards):
        print(' ', rank)
        torch.save(output[rank], os.path.join(args.output_model_path, 'consolidated.%02d.pth' % rank))

    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decompose/recompose llama model in different number of shards")
    parser.add_argument("-n","--num_shards", type=int, help="Number of new shards")
    parser.add_argument("-i","--input_model_path", type=str, help="Input model directory path")
    parser.add_argument("-o","--output_model_path", type=str, help="Output model directory path")

    args = parser.parse_args()

    main(args)
