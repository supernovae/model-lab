"""Built-in normalized annotation entry points for inspection and audit."""

from klab.corpus.language_detection import annotate_detected_language
from klab.corpus.privacy_candidates import annotate_privacy_candidates
from klab.corpus.privacy_exclusions import annotate_privacy_exclusions
from klab.corpus.source_language import annotate_source_language
from klab.corpus.source_quality import annotate_source_quality
from klab.corpus.template_lines import annotate_template_lines
from klab.corpus.text_statistics import annotate_normalized

ANNOTATORS = {
    "text-statistics": annotate_normalized,
    "source-language": annotate_source_language,
    "source-quality": annotate_source_quality,
    "privacy-exclusions": annotate_privacy_exclusions,
    "privacy-candidates": annotate_privacy_candidates,
    "detected-language": annotate_detected_language,
    "template-lines": annotate_template_lines,
}
