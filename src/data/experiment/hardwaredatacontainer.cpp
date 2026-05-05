#include "hardwaredatacontainer.h"

#include <QFile>
#include <QTextStream>
#include <QDir>
#include <QDebug>

#include <data/storage/blackchirpcsv.h>

namespace BC::Data {

bool HardwareDataContainer::saveToFile(const QString& filePath) const
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Could not open hardware file for writing:" << filePath;
        return false;
    }

    QTextStream stream(&file);

    BlackchirpCSV::writeLine(stream, {"key", "driver"});

    for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
        BlackchirpCSV::writeLine(stream, {
            it.key(),
            it.value().implementation
        });
    }

    return true;
}

HardwareDataContainer HardwareDataContainer::loadFromFile(const QString& filePath)
{
    HardwareDataContainer container;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Could not open hardware file for reading:" << filePath;
        return container;
    }

    BlackchirpCSV csv;
    bool headerSkipped = false;

    while (!file.atEnd()) {
        auto line = csv.readLine(file);
        if (line.isEmpty())
            continue;

        // The header row's first cell is "key" (any case). Accept either the
        // historical "subKey" label or the current "driver" label in the
        // second cell, and silently ignore a third cell that may carry the
        // dropped "hardwareType" column from transitional fixtures.
        if (!headerSkipped && line.first().toString().toLower() == "key") {
            headerSkipped = true;
            continue;
        }

        QString key = line.first().toString();
        if (key.isEmpty())
            continue;

        QString implementation;
        if (line.size() >= 2) {
            implementation = line.at(1).toString();
        } else {
            // Single-cell rows predate multiple-hardware support; the cell
            // is just the hardware-type root key.
            implementation = "virtual";
            key = key + ".default";
        }

        // The hardware type is fully recoverable from the key prefix in every
        // historical key shape: bare "hwType" (oldest), "hwType.index"
        // (mid-life), and "hwType.label" (current). legacyStringToHardwareType
        // already covers known aliases (e.g. "FtmwDigitizer" -> FtmwScope).
        auto keyParts = key.split('.');
        QString typeString = keyParts.isEmpty() ? key : keyParts.first();
        HardwareType hwType = HardwareDataContainer::legacyStringToHardwareType(typeString);
        container.hardwareMap[key] = HardwareEntry(implementation, hwType);
    }

    // Type keys are populated only for new (in-memory) experiments by
    // RuntimeHardwareConfig::createHardwareDataContainer; loaded experiments
    // leave them empty by design.

    return container;
}

} // namespace BC::Data