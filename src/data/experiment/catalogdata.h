#ifndef CATALOGDATA_H
#define CATALOGDATA_H

#include <QVector>
#include <QString>
#include <QVariantMap>
#include <QSharedDataPointer>

class CatalogDataPrivate;

/**
 * @brief Data structure representing a single spectroscopic transition
 */
struct TransitionData {
    double frequency;           ///< Transition frequency in MHz
    double intensity;           ///< Transition intensity (arbitrary units)
    QString quantumNumbers;     ///< Formatted quantum number assignments
    QVariantMap additionalData; ///< Format-specific additional data
    
    TransitionData() : frequency(0.0), intensity(0.0) {}
    TransitionData(double freq, double intens, const QString &quantum = QString())
        : frequency(freq), intensity(intens), quantumNumbers(quantum) {}
};

/**
 * @brief Container for spectroscopic catalog data with copy-on-write semantics
 * 
 * This class follows the Qt implicit sharing pattern used by the Ft class,
 * allowing efficient copying and memory management for large transition datasets.
 */
class CatalogData
{
public:
    CatalogData();
    CatalogData(const CatalogData &other);
    CatalogData &operator=(const CatalogData &other);
    ~CatalogData();
    
    // Data access
    QVector<TransitionData> transitions() const;
    void setTransitions(const QVector<TransitionData> &transitions);
    void addTransition(const TransitionData &transition);
    void addTransition(double frequency, double intensity, const QString &quantumNumbers = QString());
    void clear();
    
    // Metadata access
    QString sourceProgram() const;
    void setSourceProgram(const QString &program);
    
    QString moleculeName() const;
    void setMoleculeName(const QString &name);
    
    QVariantMap metadata() const;
    void setMetadata(const QVariantMap &metadata);
    void setMetadataValue(const QString &key, const QVariant &value);
    QVariant metadataValue(const QString &key, const QVariant &defaultValue = QVariant()) const;
    
    // Utility methods
    int size() const;
    bool isEmpty() const;
    TransitionData at(int index) const;
    
    // Frequency range operations
    CatalogData filterByFrequencyRange(double minFreq, double maxFreq) const;
    std::pair<double, double> frequencyRange() const;
    
    // Equality comparison
    bool operator==(const CatalogData &other) const;
    bool operator!=(const CatalogData &other) const;
    
private:
    QSharedDataPointer<CatalogDataPrivate> d;
};

#endif // CATALOGDATA_H