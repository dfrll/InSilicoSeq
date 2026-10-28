from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pysam
from pathlib import Path

from iss.generator import write_alignments


def make_read(read_id, sequence, qualities):
    read = SeqRecord(Seq(sequence), id=read_id)
    read.letter_annotations["phred_quality"] = qualities
    return read


def test_write_alignments_preserves_cigars_and_pair_fields(tmp_path):
    sam_path = tmp_path / "test.sam"
    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": "ref", "LN": 1000}],
    }

    forward = make_read("pair/1", "ACGTA", [30, 31, 32, 33, 34])
    reverse = make_read("pair/2", "TGCAT", [35, 36, 37, 38, 39])

    # forward: 2M1I2M => 5 query bases, 4 reference bases
    forward_cigar = [(0, 2), (1, 1), (0, 2)]

    # reverse: 2M1D3M => 5 query bases, 6 reference bases
    reverse_cigar = [(0, 2), (2, 1), (0, 3)]

    # header, forward_start, forward_end, reverse_start, reverse_end
    coordinates = ("ref", 100, 104, 104, 110)

    with pysam.AlignmentFile(str(sam_path), "w", header=header) as sam:
        write_alignments(
            forward,
            reverse,
            coordinates,
            forward_cigar,
            reverse_cigar,
            sam,
        )

    with pysam.AlignmentFile(str(sam_path), "r") as sam:
        records = list(sam.fetch(until_eof=True))

    assert len(records) == 2

    read1 = next(record for record in records if record.is_read1)
    read2 = next(record for record in records if record.is_read2)

    # Query name, sequence, qualities, and CIGAR survive SAM writing.
    assert read1.query_name == read2.query_name == "pair"
    assert read1.query_sequence == "ACGTA"
    assert read2.query_sequence == "TGCAT"
    assert read1.query_qualities.tolist() == [30, 31, 32, 33, 34]
    assert read2.query_qualities.tolist() == [35, 36, 37, 38, 39]
    assert read1.cigartuples == forward_cigar
    assert read2.cigartuples == reverse_cigar

    # Pair flags, coordinates, mate coordinates, and template lengths.
    assert read1.flag == 99
    assert read2.flag == 147
    assert read1.reference_start == 100
    assert read2.reference_start == 104
    assert read1.next_reference_start == read2.reference_start
    assert read2.next_reference_start == read1.reference_start
    assert read1.template_length == 10
    assert read2.template_length == -10

    # CIGAR query spans equal the sequence lengths; reference spans reflect
    # the insertion/deletion behavior.
    query_ops = {0, 1, 4, 7, 8}
    reference_ops = {0, 2, 3, 7, 8}

    for record in (read1, read2):
        query_span = sum(
            length
            for op, length in record.cigartuples
            if op in query_ops
        )
        reference_span = sum(
            length
            for op, length in record.cigartuples
            if op in reference_ops
        )

        assert query_span == len(record.query_sequence)
        assert reference_span == record.reference_length

    assert read1.reference_length == 4  # insertion doesn't consume reference
    assert read2.reference_length == 6  # deletion does consume reference
