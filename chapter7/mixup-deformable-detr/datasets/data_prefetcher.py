# # ------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------

# import torch

# def to_cuda(samples, targets, device):
#     samples = samples.to(device, non_blocking=True)
#     targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
#     return samples, targets

# class data_prefetcher():
#     def __init__(self, loader, device, prefetch=True):
#         self.loader = iter(loader)
#         self.prefetch = prefetch
#         self.device = device
#         if prefetch:
#             self.stream = torch.cuda.Stream()
#             self.preload()

#     def preload(self):
#         try:
#             self.next_samples, self.next_targets = next(self.loader)
#         except StopIteration:
#             self.next_samples = None
#             self.next_targets = None
#             return
#         # if record_stream() doesn't work, another option is to make sure device inputs are created
#         # on the main stream.
#         # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
#         # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
#         # Need to make sure the memory allocated for next_* is not still in use by the main stream
#         # at the time we start copying to next_*:
#         # self.stream.wait_stream(torch.cuda.current_stream())
#         with torch.cuda.stream(self.stream):
#             self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
#             # more code for the alternative if record_stream() doesn't work:
#             # copy_ will record the use of the pinned source tensor in this side stream.
#             # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
#             # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
#             # self.next_input = self.next_input_gpu
#             # self.next_target = self.next_target_gpu

#             # With Amp, it isn't necessary to manually convert data to half.
#             # if args.fp16:
#             #     self.next_input = self.next_input.half()
#             # else:

#     def next(self):
#         if self.prefetch:
#             torch.cuda.current_stream().wait_stream(self.stream)
#             samples = self.next_samples
#             targets = self.next_targets
#             if samples is not None:
#                 samples.record_stream(torch.cuda.current_stream())
#             if targets is not None:
#                 for t in targets:
#                     for k, v in t.items():
#                         v.record_stream(torch.cuda.current_stream())
#             self.preload()
#         else:
#             try:
#                 samples, targets = next(self.loader)
#                 samples, targets = to_cuda(samples, targets, self.device)
#             except StopIteration:
#                 samples = None
#                 targets = None
#         return samples, targets



# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, lam, targets_ground, targets_aerial, device):
    samples = samples.to(device, non_blocking=True)
    lam = torch.tensor(lam).to(device, non_blocking=True)     
    targets_ground = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_ground]
    targets_aerial = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_aerial]
    return samples, lam, targets_ground, targets_aerial

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_lam, self.next_targets_ground, self.next_targets_aerial = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets_ground = None
            self.next_targets_aerial = None
            self.next_lam = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_lam, self.next_targets_ground, self.next_targets_aerial = to_cuda(self.next_samples, self.next_lam, self.next_targets_ground, self.next_targets_aerial, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:


    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets_ground = self.next_targets_ground
            targets_aerial = self.next_targets_aerial
            lam = self.next_lam
            #targets = self.next_targets
            
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())

            if targets_ground is not None:
                for t in targets_ground:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())

            if targets_aerial is not None:
                for t in targets_aerial:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())

            if lam is not None:
                torch.tensor(lam).record_stream(torch.cuda.current_stream())

            self.preload()
        else:
            try:
                samples, lam, targets_ground, targets_aerial = next(self.loader) # samples, {"ground": ground_target, "aerial": aerial_targets}
                samples, lam, targets_ground, targets_aerial = to_cuda(samples, lam, targets_ground, targets_aerial, self.device)
            except StopIteration:
                samples = None
                targets_ground = None
                targets_aerial = None
                lam = None
        return samples, lam, targets_ground, targets_aerial
