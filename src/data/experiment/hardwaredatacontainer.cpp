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
    
    // Write header row (NEW 3-column format with hardware type enum)
    BlackchirpCSV::writeLine(stream, {"key", "subKey", "hardwareType"});
    
    // Write hardware selections in new format
    for (auto it = hardwareMap.begin(); it != hardwareMap.end(); ++it) {
        BlackchirpCSV::writeLine(stream, {
            it.key(), 
            it.value().implementation, 
            static_cast<int>(it.value().type)
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
        return container; // Return empty container
    }
    
    // Create a CSV reader to parse the file
    BlackchirpCSV csv;
    
    bool headerSkipped = false;
    bool isNewFormat = false; // Track whether we're reading new 3-column format
    
    while (!file.atEnd()) {
        auto line = csv.readLine(file);
        
        // Skip header row and detect format
        if (!headerSkipped) {
            if (line.size() >= 2 && line.first().toString().toLower() == "key") {
                // Check if this is new 3-column format
                isNewFormat = (line.size() == 3 && line.last().toString().toLower().contains("hardwaretype"));
                headerSkipped = true;
                continue;
            }
        }
        
        // Parse data rows based on format
        if (isNewFormat) {
            // New 3-column format: key, implementation, hardwareType
            if (line.size() != 3) {
                continue; // Skip malformed lines
            }
            
            QString key = line[0].toString();
            QString implementation = line[1].toString();
            int hardwareTypeInt = line[2].toInt();
            
            // Skip empty keys
            if (key.isEmpty()) {
                continue;
            }
            
            HardwareType hwType = static_cast<HardwareType>(hardwareTypeInt);
            container.hardwareMap[key] = HardwareEntry(implementation, hwType);
            
        } else if (line.size() == 2) {
            // Legacy 2-column format: key, implementation
            QString key = line[0].toString();
            QString implementation = line[1].toString();
            
            // Skip empty keys or the header row if we missed it
            if (key.isEmpty() || key.toLower() == "key") {
                continue;
            }
            
            // Extract hardware type string from key and use legacy lookup
            auto keyParts = key.split('.');
            QString typeString = keyParts.isEmpty() ? key : keyParts.first();
            HardwareType hwType = HardwareDataContainer::legacyStringToHardwareType(typeString);
            container.hardwareMap[key] = HardwareEntry(implementation, hwType);
            
        } else if (line.size() == 1) {
            // Very old format: just hardware type, no multiple hardware support
            QString typeString = line[0].toString();
            
            // Skip empty types or header rows
            if (typeString.isEmpty() || typeString.toLower() == "key") {
                continue;
            }
            
            // Create a dummy key with default label for very old format
            QString key = typeString + ".default";
            QString implementation = "virtual"; // Default to virtual implementation
            HardwareType hwType = HardwareDataContainer::legacyStringToHardwareType(typeString);
            container.hardwareMap[key] = HardwareEntry(implementation, hwType);
        }
        // Skip lines that don't match any expected format
    }
    
    // Type keys remain empty for loaded experiments (only populated for new experiments)
    // This is the desired behavior as noted in the class documentation
    
    return container;
}

} // namespace BC::Data