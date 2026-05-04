#ifndef SPCATPARSER_H
#define SPCATPARSER_H

#include "catalogparser.h"

/**
 * @brief Parser for SPCAT catalog files (.cat)
 * 
 * SPCAT format specification:
 * - Fixed-width 80-character lines
 * - Frequency (MHz), Error (MHz), Intensity (log10), Degeneracy, Lower energy (cm⁻¹)
 * - Species tag, format code, quantum number assignments
 * - Quantum numbers start at position 55, format varies by molecule type
 */
class SPCATParser : public CatalogParser
{
public:
    SPCATParser();
    
    // CatalogParser interface
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;
    CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;
    QString formatName() const override;
    QString formatDescription() const override;
    QStringList fileExtensions() const override;

private:
    /**
     * @brief Parse a single line from SPCAT catalog
     * @param line 80-character fixed-width line
     * @return TransitionData structure, or invalid data if parsing fails
     */
    TransitionData parseLine(const QString &line) const;
    
    /**
     * @brief Extract and format quantum numbers from SPCAT line
     * @param line Full catalog line
     * @param formatCode SPCAT format code for quantum number interpretation
     * @return Formatted quantum number string with semicolons removed
     */
    QString parseQuantumNumbers(const QString &line, int formatCode) const;
    
    /**
     * @brief Convert SPCAT log intensity to linear scale
     * @param logIntensity log10(intensity in nm²MHz)
     * @return Linear intensity value
     */
    double convertIntensity(double logIntensity) const;
    
    /**
     * @brief Validate SPCAT file format by checking line structure
     * @param filePath Path to potential SPCAT file
     * @return true if file appears to be valid SPCAT format
     */
    bool validateFormat(const QString &filePath) const;
};

#endif // SPCATPARSER_H