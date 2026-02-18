import torch
from crane import BrainFeatureExtractor
from crane.core.featurizer import BrainFeature
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

    def forward(self, batch: Data) -> BrainFeature:
        """
        Args:
            batch: Data object containing raw iEEG data

        Returns:
            Spectrogram features: [batch_size, num_electrodes, num_timebins, n_freqs]
        """
        ieeg, channels = batch.data.data, batch.channels.id  # type: ignore[attr-defined]
        sampling_rate = batch.data.sampling_rate  # type: ignore[attr-defined]
        ieeg = torch.from_numpy(ieeg.T).float()
        if self.rereference:
            ieeg, channels = laplacian_rereference(ieeg, channels)
        ieeg, _ = subset_electrodes(ieeg, channels, self.max_n_electrodes)

        # Add dummy batch dimension for spectrogram computation
        ieeg = ieeg.unsqueeze(0)  # [1, num_electrodes, num_timepoints]
        spec = self.spectrogram(ieeg, sampling_rate)  # [1, num_electrodes, num_timebins, n_freqs]
        return BrainFeature(spec=spec.squeeze(0))  # [num_electrodes, num_timebins, n_freqs]
