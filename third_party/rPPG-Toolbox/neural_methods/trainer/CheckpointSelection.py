"""Record validation-only selection once, before any held-out test is run."""
import hashlib
import json
import math
import os
import re
from pathlib import Path


def register_best_checkpoint(config, trainer):
    if config.TEST.USE_LAST_EPOCH:
        return None
    epoch, loss = int(trainer.best_epoch), float(trainer.min_valid_loss)
    if not 0 <= epoch < config.TRAIN.EPOCHS or not math.isfinite(loss):
        raise ValueError('Validation selection has an invalid epoch or non-finite loss')
    name = config.TRAIN.MODEL_FILE_NAME
    source = (Path(trainer.model_dir) / f'{name}_Epoch{epoch}.pth').resolve()
    h = hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    aliases = [source.with_name(f'{name}_Best.pth')]
    match = re.fullmatch(r'KFold_(physnet|physformer|deepphys|tscan|efficientphys)_fold([0-4])', name)
    if match and config.TRAIN.DATA.DATASET == 'KoreanrPPG':
        checkpoint_root = os.environ.get('KR_RPPG_CHECKPOINTS')
        if not checkpoint_root:
            raise ValueError('Set KR_RPPG_CHECKPOINTS to register KR validation-best checkpoints')
        aliases.append(Path(checkpoint_root) / 'kr_5fold' / match[1] / f'{name}_Best.pth')
    for alias in aliases:
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != source:
                raise ValueError(f'Existing Best alias belongs to a different epoch: {alias}')
        else:
            alias.symlink_to(os.path.relpath(source, alias.parent))
    record = {'policy': 'minimum_validation_loss_first_tie', 'epoch': epoch, 'validation_loss': loss,
              'checkpoint': str(source), 'checkpoint_sha256': h.hexdigest(),
              'aliases': [str(p) for p in aliases], 'test_results_used_for_selection': False,
              'expanded_config_sha256': hashlib.sha256(config.dump().encode()).hexdigest()}
    if match:
        record.update(model=match[1], fold=int(match[2]))
    destination = Path(config.LOG.PATH) / 'selected_checkpoint.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(destination)
    print(f'[checkpoint-selection] {destination}: epoch {epoch}, validation loss {loss}', flush=True)
    return record
