#ifndef XIAMPARSER_H
#define XIAMPARSER_H

#include "catalogparser.h"
#include <QRegularExpression>

/// \brief Parser for XIAM (eXtended Internal Axis Method) catalog output.
class XIAMParser : public CatalogParser
{
public:
    /// \brief Recognize a file by its ``.xo`` or ``.out`` suffix and the
    /// XIAM ``-- B`` header pattern.
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;

    /// \brief Parse a recognized XIAM file into a :cpp:class:`CatalogData`.
    ///
    /// Detects the intensity mode (``ints=2`` or ``ints=3``) automatically
    /// and dispatches to the matching internal parser.
    CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;

    /// \brief Returns ``"XIAM"``.
    QString formatName() const override;

    /// \brief Returns a one-line description of the format.
    QString formatDescription() const override;

    /// \brief Returns ``{"*.xo", "*.out"}``.
    QStringList fileExtensions() const override;

private:
    /// \brief Detect the XIAM intensity mode used in the file.
    /// \return ``2`` for ``ints=2``, ``3`` for ``ints=3``, ``0`` if no
    ///         mode declaration is found.
    int detectIntensityMode(const QStringList &lines) const;

    /// \brief Parse the ``ints=2`` (simple) variant.
    CatalogData parseInts2Format(const QStringList &lines, int startLine) const;

    /// \brief Parse the ``ints=3`` (with splitting analysis) variant.
    CatalogData parseInts3Format(const QStringList &lines, int startLine) const;

    /// \brief Find the line index where transition data begins.
    /// \return Zero-based line index, or ``-1`` if no header is found.
    int findDataStartLine(const QStringList &lines) const;

    /// \brief Normalize a XIAM quantum-number string.
    QString parseQuantumNumbers(const QString &qnString) const;

    /// \brief Extract the molecule name from the file header.
    /// \return Header-supplied name when present; otherwise the file's
    ///         base name as a fallback.
    QString extractMoleculeName(const QStringList &lines, const QString &filePath) const;

    /// \brief Parse a single transition line in ``ints=2`` mode.
    /// \param line Raw transition line from the XIAM output.
    /// \param blockNumber Block label appended to the quantum-number string when the file contains multiple blocks.
    TransitionData parseInts2Line(const QString &line, const QString &blockNumber = QString()) const;

    /// \brief Parse a single transition line in ``ints=3`` mode.
    /// \param line Raw transition line from the XIAM output.
    /// \param groupQuantumNumbers Quantum numbers carried over from the block start so split lines inherit the parent assignment.
    TransitionData parseInts3Line(const QString &line, const QString &groupQuantumNumbers = QString()) const;

    /// \brief Reconstruct a high-precision intensity from XIAM's
    /// constituent fields.
    ///
    /// XIAM prints intensities with fixed decimal precision; for weak
    /// transitions the printed total can lose significant digits.
    /// This helper recomputes the intensity from line strength,
    /// statistical weight, population, and energy factors and returns
    /// whichever value is more precise.
    double calculateOptimalIntensity(double linestr, double total, double statWeight,
                                     double population, double hvEnergy) const;
};

#endif // XIAMPARSER_H
