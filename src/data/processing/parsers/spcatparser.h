#ifndef SPCATPARSER_H
#define SPCATPARSER_H

#include "catalogparser.h"

/// \brief Parser for SPCAT catalog (``*.cat``) output.
class SPCATParser : public CatalogParser
{
public:
    SPCATParser();

    /// \brief Recognize a file by ``.cat`` suffix and a structural sniff
    /// of the first lines.
    bool canParse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;

    /// \brief Parse a recognized SPCAT file into a :cpp:class:`CatalogData`.
    ///
    /// Sets the ``sourceProgram`` field to ``"SPCAT"`` and the
    /// ``moleculeName`` field to the file's base name. Lines that do not
    /// produce a valid transition are silently skipped.
    CatalogData parse(const QString &filePath, const QVariantMap &hints = QVariantMap()) const override;

    /// \brief Returns ``"SPCAT"``.
    QString formatName() const override;

    /// \brief Returns a one-line description of the format.
    QString formatDescription() const override;

    /// \brief Returns ``{"*.cat"}``.
    QStringList fileExtensions() const override;

private:
    /// \brief Parse a single 80-character SPCAT line into a TransitionData.
    /// \return TransitionData with positive frequency on success; an
    ///         invalid (zero-frequency) record on failure.
    TransitionData parseLine(const QString &line) const;

    /// \brief Extract and format the quantum-number block from an SPCAT
    /// line.
    /// \param line Full 80-character catalog line.
    /// \param formatCode SPCAT format code that controls the quantum-number layout.
    /// \return Formatted quantum-number string with embedded semicolons
    ///         stripped for CSV safety.
    QString parseQuantumNumbers(const QString &line, int formatCode) const;

    /// \brief Convert a base-10 log intensity to its linear value.
    /// \param logIntensity log₁₀(intensity in nm²·MHz).
    double convertIntensity(double logIntensity) const;

    /// \brief Sniff the first lines of ``filePath`` to confirm SPCAT
    /// fixed-width structure.
    bool validateFormat(const QString &filePath) const;
};

#endif // SPCATPARSER_H
