#ifndef XIAMPARSER_H
#define XIAMPARSER_H

#include "catalogparser.h"
#include <QRegularExpression>

/**
 * @brief Parser for XIAM spectroscopic catalog output files
 * 
 * XIAM (eXtended Internal Axis Method) is a program for analyzing
 * internal rotation effects in molecular spectra. This parser handles
 * both ints=2 (simple) and ints=3 (with splitting analysis) output modes.
 * 
 * Format characteristics:
 * - ints=2: Direct frequency and intensity listings
 * - ints=3: Includes rigid rotor reference and symmetry state splittings
 * - File extensions: .xo (output), .out
 * - Header pattern: "-- B [num]" followed by column headers
 */
class XIAMParser : public CatalogParser
{
public:
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;
    CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;
    QString formatName() const override;
    QString formatDescription() const override;
    QStringList fileExtensions() const override;

private:
    /**
     * @brief Detect which XIAM intensity mode is used in the file
     * @param lines File content lines
     * @return 2 for ints=2 mode, 3 for ints=3 mode, 0 if unknown
     */
    int detectIntensityMode(const QStringList &lines) const;
    
    /**
     * @brief Parse XIAM file in ints=2 mode (simple format)
     * @param lines File content lines
     * @param startLine Index where data section begins
     * @return CatalogData with parsed transitions
     */
    CatalogData parseInts2Format(const QStringList &lines, int startLine) const;
    
    /**
     * @brief Parse XIAM file in ints=3 mode (with splitting analysis)
     * @param lines File content lines  
     * @param startLine Index where data section begins
     * @return CatalogData with parsed transitions
     */
    CatalogData parseInts3Format(const QStringList &lines, int startLine) const;
    
    /**
     * @brief Find the line where transition data begins
     * @param lines File content lines
     * @return Line index, or -1 if not found
     */
    int findDataStartLine(const QStringList &lines) const;
    
    /**
     * @brief Parse quantum number assignments from XIAM format
     * @param qnString Quantum number string (e.g., "K -1  0  t  2  1")
     * @return Formatted quantum number string
     */
    QString parseQuantumNumbers(const QString &qnString) const;
    
    /**
     * @brief Extract molecule name from file header
     * @param lines File content lines
     * @return Molecule name or filename if not found
     */
    QString extractMoleculeName(const QStringList &lines, const QString &filePath) const;
    
    /**
     * @brief Parse a transition line in ints=2 format
     * @param line Input line containing transition data
     * @param blockNumber Block number string to append to quantum numbers
     * @return TransitionData structure, or invalid transition if parsing fails
     */
    TransitionData parseInts2Line(const QString &line, const QString &blockNumber = QString()) const;
    
    /**
     * @brief Parse an individual transition line in ints=3 format
     * @param line Input line
     * @param groupQuantumNumbers Quantum numbers from group start (for split lines)
     * @return TransitionData structure
     */
    TransitionData parseInts3Line(const QString &line, const QString &groupQuantumNumbers = QString()) const;
    
    /**
     * @brief Calculate optimal intensity considering XIAM's fixed decimal precision issues
     * @param linestr Line strength from XIAM output
     * @param total Total intensity from XIAM output
     * @param statWeight Statistical weight
     * @param population Population factor
     * @param hvEnergy Energy factor
     * @return Optimal intensity value (either linestr or calculated from components)
     */
    double calculateOptimalIntensity(double linestr, double total, double statWeight, double population, double hvEnergy) const;
};

#endif // XIAMPARSER_H