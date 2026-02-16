import torch
from crane import BrainFeatureExtractor
from crane.preprocess import Spectrogram, laplacian_rereference, subset_electrodes
from temporaldata import Data


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

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Args:
            batch: Data object containing raw iEEG data

        Returns:
            Spectrogram features: [batch_size, num_electrodes, num_timebins, n_freqs]
        """

        ieeg, channels = batch.data, batch.channels.id  # type: ignore[attr-defined]
        if self.rereference:
            ieeg, channels = laplacian_rereference(ieeg.data, channels)
        ieeg, _ = subset_electrodes(ieeg, channels, self.max_n_electrodes)

        return self.spectrogram(ieeg, ieeg.sampling_rate)
