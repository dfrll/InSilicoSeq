#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import sys

import _pickle
import numpy as np
from itertools import groupby
from Bio.Seq import MutableSeq, Seq

from iss import util


class ErrorModel(object):
    """Main ErrorModel Class

    This class is used to create inheriting classes and contains all
    the functions that are shared by all ErrorModel classes
    """

    @property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        return logging.getLogger(component)

    def load_npz(self, npz_path, model):
        """load the error profile .npz file

        Args:
            npz_path (string): path to the npz file
            model (string): type of model. Could be 'cdf' or 'kde'. 'cdf' has
                been deprecated and is no longer available

        Returns:
            ndarray: numpy object containg variables necessary
                for error model construction
        """
        try:
            error_profile = np.load(npz_path, allow_pickle=True)
            assert error_profile["model"] == model
        except (OSError, IOError, EOFError, _pickle.UnpicklingError) as e:
            self.logger.error("Failed to read ErrorModel file: %s" % e)
            sys.exit(1)
        except AssertionError:
            self.logger.error("Trying to load a %s ErrorModel in %s mode" % (error_profile["model"], model))
            sys.exit(1)
        else:
            self.logger.debug("Loaded ErrorProfile: %s" % npz_path)
        return error_profile

    def introduce_error_scores(self, record, orientation):
        """Add phred scores to a SeqRecord according to the error_model

        Args:
            record (SeqRecord): a read record
            orientation (string): orientation of the read. Can be 'forward' or
                'reverse'

        Returns:
            SeqRecord: a read record with error scores
        """
        if orientation == "forward":
            record.letter_annotations["phred_quality"] = self.gen_phred_scores(self.quality_forward, "forward")
        elif orientation == "reverse":
            record.letter_annotations["phred_quality"] = self.gen_phred_scores(self.quality_reverse, "reverse")
        return record

    def mut_sequence(self, record, orientation):
        """Introduce substitution errors to a sequence

        If a random probability is higher than the probability of the basecall
        being correct, introduce a substitution error

        Args:
            record (SeqRecord): a read record with error scores
            orientation (string): orientation of the read. Can be 'forward' or
                'reverse'

        Returns:
            SeqRecord: the read record with substitution errors
        """

        # get the right subst_matrix
        if orientation == "forward":
            nucl_choices = self.subst_choices_for
        elif orientation == "reverse":
            nucl_choices = self.subst_choices_rev

        mutable_seq = MutableSeq(record.seq)
        quality_list = record.letter_annotations["phred_quality"]
        position = 0
        for nucl, qual in zip(mutable_seq, quality_list):
            if random.random() > util.phred_to_prob(qual) and nucl.upper() not in "RYWSMKHBVDN":
                mutated_nuc = str(
                    np.random.choice(nucl_choices[position][nucl.upper()][0], p=nucl_choices[position][nucl.upper()][1])
                )
                if self.store_mutations and mutated_nuc != record.annotations["original"][position]:
                    record.annotations["mutations"].append(
                        {
                            "id": record.id,
                            "position": position,
                            "ref": mutable_seq[position],
                            "alt": mutated_nuc,
                            "quality": qual,
                            "type": "sub",
                        }
                    )
                mutable_seq[position] = mutated_nuc
            position += 1
        record.seq = Seq(mutable_seq)
        return record

    def introduce_indels(self, record, orientation, full_seq, bounds):
        """Introduce insertions or deletions in a sequence

        Introduce insertion and deletion errors according to the probabilities
        present in the indel choices list


        Args:
            record (SeqRecord): a sequence record
            orientation (string): orientation of the read. Can be 'forward' or
                'reverse'
            full_seq (Seq): the reference sequence from which mut_seq
                comes from
            bounds (tuple): the position of the read in the full_sequence

        Returns:
            SeqRecord: a sequence record with indel errors
        """

        # get the right indel arrays
        if orientation == "forward":
            insertions = self.ins_for
            deletions = self.del_for
        elif orientation == "reverse":
            insertions = self.ins_rev
            deletions = self.del_rev
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        original = str(record.seq)
        read_start, read_end = bounds

        qualities = record.letter_annotations.get("phred_quality")
        if qualities is None:
            raise ValueError(f"{record.id} has no phred_quality scores")

        qualities = list(qualities)

        sequence = []
        new_qualities = []
        ops = []

        def add_op(op):
            ops.append((op, 1))

        for position, base in enumerate(original):
            ambiguous = base.upper() in "RYWSMKHBVDN"

            deleted = (
                not ambiguous
                and position < len(deletions)
                and random.random() < deletions[position].get(base.upper(), 0)
            )

            if deleted:
                add_op(2) # D: consumes reference, not query
                if self.store_mutations:
                    record.annotations.setdefault("mutations", []).append({
                       "id": record.id,
                       "position": position,
                       "ref": base,
                       "alt": ".",
                       "quality": ".",
                       "type": "del",
                    })
                continue

            sequence.append(base)
            add_op(0) # M
            new_qualities.append(qualities[position])

            if not ambiguous and position < len(insertions):
                for inserted_base, probability in insertions[position].items():
                    if random.random() < probability:
                        inserted_base = str(inserted_base)
                        sequence.append(inserted_base)
                        add_op(1) # I

                        new_qualities.append(qualities[position])

                        if self.store_mutations:
                            record.annotations.setdefault("mutations", []).append({
                                "id": record.id,
                                "position": position,
                                "ref": base,
                                "alt": base + inserted_base,
                                "quality": ".",
                                "type": "ins"
                            })

        sequence, new_qualities, ops, read_start, read_end = self._adjust_seq_length(
            sequence, new_qualities, ops, orientation, full_seq,
            read_start, read_end
        )

        record.letter_annotations.clear()
        record.seq = Seq("".join(sequence))
        record.letter_annotations["phred_quality"] = new_qualities

        return record, self._compress_cigar(ops), (read_start, read_end)


    def _adjust_seq_length(
        self, sequence, qualities, ops, orientation, full_sequence,
        read_start, read_end,
    ):
        """Adjust a read to the configured length, keeping qualities and CIGAR in sync.

        Truncate excess query bases from the end, or extend the read using bases
        from the reference. If the reference has no bases available, pad with
        ``"A"`` and represent those bases as soft clips.

        Args:
            sequence: Read bases.
            qualities: Per-base quality values.
            ops: CIGAR operations as ``(operation_code, length)`` pairs.
            orientation: Read orientation, either ``"forward"`` or ``"reverse"``.
            full_sequence: Reference sequence used to extend the read.
            read_start: Start coordinate of the read on the reference.
            read_end: End coordinate of the read on the reference.

        Returns:
            A tuple containing the adjusted sequence, qualities, CIGAR operations,
            start coordinate, and end coordinate.
        """
        sequence = list(sequence)
        qualities = list(qualities)
        ops = list(ops)

        def consumes_query(op):
            return op in (0, 1, 4, 7, 8)  # M, I, S, =, X

        if len(sequence) > self.read_length:
            excess = len(sequence) - self.read_length
            del sequence[-excess:]
            del qualities[-excess:]

            while excess and ops:
                op, length = ops[-1]
                if not consumes_query(op):
                    ops.pop()
                elif length <= excess:
                    excess -= length
                    ops.pop()
                else:
                    ops[-1] = (op, length - excess)
                    excess = 0
            
            while ops and ops[-1][0] == 2:
                ops.pop()

        elif len(sequence) < self.read_length:
            to_add = self.read_length - len(sequence)
            matches = 0
            softclips = 0

            for _ in range(to_add):
                if orientation == "forward":
                    ref_pos = read_end
                    if ref_pos < len(full_sequence):
                        sequence.append(str(full_sequence[ref_pos]))
                        read_end += 1
                        matches +=1
                    else:
                        sequence.append("A")
                        softclips += 1
                
                elif orientation == "reverse":
                    ref_pos = read_start - 1
                    if ref_pos >= 0:
                        sequence.append(str(util.rev_comp(full_sequence[ref_pos])))
                        read_start -= 1
                        matches += 1
                    else:
                        sequence.append("A")
                        softclips += 1
                else:
                    raise ValueError(f"Unknown orientation: {orientation}")
                    
                qualities.append(qualities[-1] if qualities else 0)
            
            if matches:
                ops.append((0, matches)) # M
            if softclips:
                ops.append((4, softclips)) # S

        return sequence, qualities, ops, read_start, read_end

    def _compress_cigar(self, ops):
        """Merge adjacent identical CIGAR operations.

        Args:
            ops: CIGAR operations as ``(operation_code, length)`` pairs.

        Returns:
            A list of ``(operation_code, combined_length)`` pairs.
        """
        compressed = []
        for op, length in ops:
            if compressed and compressed[-1][0] == op:
                previous_op, previous_length = compressed[-1]
                compressed[-1] = (previous_op, previous_length + length)
            else:
                compressed.append((op, length))
        return compressed
