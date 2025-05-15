from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

from getpass import getpass

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import matplotlib.pyplot as pl
import py3Dmol
import torch
from esm.sdk import client
from esm.utils.structure.protein_chain import ProteinChain

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
# login()

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda") # or "cpu"


template_gfp = ESMProtein.from_pdb("./downloads/1qy3_A.pdb")

template_gfp_tokens = model.encode(template_gfp)

print("Sequence tokens:")
print(
    "    ", ", ".join([str(token) for token in template_gfp_tokens.sequence.tolist()])
)

print("Structure tokens:")
print(
    "    ", ", ".join([str(token) for token in template_gfp_tokens.structure.tolist()])
)

'''
We'll now build a prompt. Specifically we'll specify 4 amino acid identities at positions near where we want the chromophore to form, 
and 2 amino acid identities on the beta barrel that are known to support chromophore formation.
Furthermore we'll specify the structure should be similar to the 1qy3 structure at all these positions 
by adding tokens from the encoded 1qy3 structure to the structure track of our prompt.
We'll also specify a few more positions (along the alpha helix kink).
'''

prompt_sequence = ["_"] * len(template_gfp.sequence)
prompt_sequence[59] = "T"
prompt_sequence[62] = "T"
prompt_sequence[63] = "Y"
prompt_sequence[64] = "G"
prompt_sequence[93] = "R"
prompt_sequence[219] = "E"
prompt_sequence = "".join(prompt_sequence)

print("gfp seq",template_gfp.sequence)
print("prompt seq",prompt_sequence)

prompt = model.encode(ESMProtein(sequence=prompt_sequence))

# We construct an empty structure track like |<bos> <mask> ... <mask> <eos>|...
prompt.structure = torch.full_like(prompt.sequence, 4096)
prompt.structure[0] = 4098
prompt.structure[-1] = 4097
# ... and then we fill in structure tokens at key residues near the alpha helix
# kink and at the stabilizing R and E positions on the beta barrel.
prompt.structure[55:70] = template_gfp_tokens.structure[56:71]
prompt.structure[93] = template_gfp_tokens.structure[93]
prompt.structure[219] = template_gfp_tokens.structure[219]

print("prompt str","".join(["✔" if st < 4096 else "_" for st in prompt.structure]))


'''
The output shows the original 1qy3 sequence and the our prompt sequence track amino acid identities and the positions that have a token on the structure track.
ESM3 will then be tasked with filling in the structure and sequence at the remaining masked (underscore) positions.

One small note, we introduced the mutation A93R in our prompt. This isn't a mistake. 
Using Alanine at this position causes the chromophore to mature extremely slowly (which is how we are able to measure the precyclized structure of GFP!).
However we don't want to wait around for our GFPs to glow so we go with Arginine at this position.
'''

num_tokens_to_decode = min((prompt.structure == 4096).sum().item(), 20)


structure_generation = model.generate(
    prompt,
    GenerationConfig(
        # Generate a structure.
        track="structure",
        # Sample one token per forward pass of the model.
        num_steps=num_tokens_to_decode,
        # Sampling temperature trades perplexity with diversity.
        temperature=1.0,
    ),
)

print("These are the structure tokens corresponding to our new design:")
print(
    "    ", ", ".join([str(token) for token in structure_generation.structure.tolist()])
)


# Decodes structure tokens to backbone coordinates.
structure_generation_protein = model.decode(structure_generation)

structure_generation_protein.to_pdb("./new_gfp_struct.pdb")


'''
At this point we only want to continue the generation if this design is a close match to a wildtype GFP at the active site, 
has some structural difference across the full protein (otherwise it would end up being very sequence-similar to wildtype GFP), 
and overall still looks like the classic GFP alpha helix in a beta barrel structure.

Of course when generating many designs we cannot look at each one manually, so we adopt some automated rejection sampling criteria 
based on the overall structure RMSD and the constrained site RMSD for our generated structure being faithful to the prompt. 
If these checks pass then we'll try to design a sequence for this structure. If not, one should go back up a few cells and design another structure 
until it passes these computational screens. (Or not... this is your GFP design!)
'''

constrained_site_positions = [59, 62, 63, 64, 93, 219]

template_chain = template_gfp.to_protein_chain()
generation_chain = structure_generation_protein.to_protein_chain()

constrained_site_rmsd = template_chain[constrained_site_positions].rmsd(
    generation_chain[constrained_site_positions]
)
backbone_rmsd = template_chain.rmsd(generation_chain)

c_pass = "✅" if constrained_site_rmsd < 1.5 else "❌"
b_pass = "✅" if backbone_rmsd > 1.5 else "❌"

print(f"Constrained site RMSD: {constrained_site_rmsd:.2f} Ang {c_pass}")
print(f"Backbone RMSD: {backbone_rmsd:.2f} Ang {b_pass}")



'''
Now we have a backbone with some structural variation but that also matches the GFP constrained site,
and we want to design a sequence that folds to this structure. We can use the prior generation, 
which is exactly our original prompt plus the new structure tokens representing the backbone, to prompt ESM3 again.

One we have designed a sequence we'll want to confirm that sequence is a match for our structure, 
so we'll remove all other conditioning from the prompt and fold the sequence. Conveniently with ESM3, 
folding a sequence is simply generating a set of structure tokens conditioned on the amino acid sequence. 
In this case we want the model's highest confidence generation (with no diversity) so we sample with a temperature of zero.
'''

# Based on internal research, there's not a benefit to iterative decoding past 20 steps
num_tokens_to_decode = min((prompt.sequence == 32).sum().item(), 20)

sequence_generation = model.generate(
    # Generate a sequence.
    structure_generation,
    GenerationConfig(track="sequence", num_steps=num_tokens_to_decode, temperature=1.0),
)

# Refold
sequence_generation.structure = None
length_of_sequence = sequence_generation.sequence.numel() - 2
sequence_generation = model.generate(
    sequence_generation,
    GenerationConfig(track="structure", num_steps=1, temperature=0.0),
)

# Decode to AA string and coordinates.
sequence_generation_protein = model.decode(sequence_generation)


print("designed gfp seq",sequence_generation_protein.sequence)

'''
We can align this sequence against the original template to see how similar it is to avGFP. 
One might also want to search against all known fluorescent proteins to assess the novelty of this potential GFP.
'''

seq1 = seq.ProteinSequence(template_gfp.sequence)
seq2 = seq.ProteinSequence(sequence_generation_protein.sequence)

alignments = align.align_optimal(
    seq1, seq2, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1)
)

alignment = alignments[0]

identity = align.get_sequence_identity(alignment)
print(f"Sequence identity: {100*identity:.2f}%")

print("\nSequence alignment:")
fig = pl.figure(figsize=(8.0, 4.0))
ax = fig.add_subplot(111)
graphics.plot_alignment_similarity_based(
    ax, alignment, symbols_per_line=45, spacing=2, show_numbers=True
)
fig.tight_layout()
pl.savefig("alignment.png", dpi=300)

'''
We now recheck our computational metrics for the constrained site. If we see the constrained site is not a match 
then we'd want to try designing the sequence again. If many attempts to design a sequence that matches the structure fail, 
then it's likely the structure is not easily designable and we may want to reject this structure generation as well!

At this point the backbone RMSD doesn't matter very much to us, so long as the sequence is adequately distant to satisfy our scientific curiosity!
'''

template_chain = template_gfp.to_protein_chain()
generation_chain = sequence_generation_protein.to_protein_chain()

constrained_site_rmsd = template_chain[constrained_site_positions].rmsd(
    generation_chain[constrained_site_positions]
)
backbone_rmsd = template_chain.rmsd(generation_chain)

c_pass = "✅" if constrained_site_rmsd < 1.5 else "❌"
b_pass = "🤷‍♂️"

print(f"Constrained site RMSD: {constrained_site_rmsd:.2f} Ang {c_pass}")
print(f"Backbone RMSD: {backbone_rmsd:.2f} Ang {b_pass}")


sequence_generation_protein.to_pdb("./new_gfp_seq_struct.pdb")