import torch
from crane import BrainFeature, BrainFeatureExtractor, CraneFeature
from crane.preprocess import Spectrogram, laplacian_rereference, subset_electrodes
from jaxtyping import Float


class SpectrogramFeature(BrainFeature):
    signals: Float[torch.Tensor, "[batch_size] num_electrodes num_timebins n_freqs"]
    """Spectrogram features computed from iEEG data, with shape (batch_size, num_electrodes, num_timebins, n_freqs)"""


class iEEGPreprocessor(BrainFeatureExtractor):
    """Preprocessor for intracranial EEG (iEEG) data that computes spectrogram features.

    Args:
        spectrogram: Spectrogram object to compute spectrogram features.
        rereference: Whether to apply laplacian rereferencing to the iEEG data.
        max_n_electrodes: Maximum number of electrodes to consider. Excess electrodes will be discarded.
    """

    def __init__(self, spectrogram: Spectrogram, rereference: bool = True, max_n_electrodes: int = 64, **kwargs):
        feature_size = spectrogram.n_freqs
        super().__init__(
            feature_size=feature_size,
            sampling_rate=-1,  # Set dynamically based on input data
            padding_value=0.0,
            **kwargs,
        )

        self.spectrogram = spectrogram
        self.rereference = rereference
        self.max_n_electrodes = max_n_electrodes

    def forward(self, batch: CraneFeature) -> SpectrogramFeature:
        """
        Args:
            batch: Data object containing raw iEEG data

        Returns:
            Spectrogram features: [batch_size, num_electrodes, num_timebins, n_freqs]
        """

        if self.rereference:
            batch = laplacian_rereference(batch)
        batch = subset_electrodes(batch, max_n_electrodes=self.max_n_electrodes)

        # Add dummy batch dimension for spectrogram computation
        signals = batch.signals.unsqueeze(0)  # [1, num_electrodes, num_timepoints]
        spec = self.spectrogram(signals, batch.sampling_rate)  # [1, num_electrodes, num_timebins, n_freqs]

        return SpectrogramFeature(signals=spec.squeeze(0))  # [num_electrodes, num_timebins, n_freqs]
