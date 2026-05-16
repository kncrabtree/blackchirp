#ifndef PEAKFINDSETTINGS_H
#define PEAKFINDSETTINGS_H

/*!
 \brief Plain aggregate of the peak-search parameters that an
        experiment remembers independently of the global defaults.

 Mirrors the per-experiment persistence model used for FID processing
 settings: the display-filter grid and the "In view" toggle are global
 view state and are deliberately not part of this struct.
*/
struct PeakFindSettings
{
    double minFreq;
    double maxFreq;
    double snr;
    double navHalfWidth;
    int winSize;
    int polyOrder;
};

#endif // PEAKFINDSETTINGS_H
