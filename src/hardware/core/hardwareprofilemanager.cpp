#include "hardwareprofilemanager.h"
#include "hardwareregistry.h"
#include "runtimehardwareconfig.h"

#include <QDateTime>
#include <QDataStream>
#include <QByteArray>
#include <QIODevice>
#include <QDebug>
#include <QRegularExpression>
#include <QReadLocker>
#include <QWriteLocker>

#include <hardware/core/ftmwdigitizer/virtualftmwscope.h>
#include <hardware/core/clock/fixedclock.h>
#include <hardware/core/lifdigitizer/virtuallifscope.h>
#include <hardware/core/liflaser/virtualliflaser.h>
#include <data/storage/applicationconfigmanager.h>

// Static member definitions
HardwareProfileManager* HardwareProfileManager::s_instance = nullptr;

// ========================================================================
// SINGLETON MANAGEMENT
// ========================================================================

HardwareProfileManager& HardwareProfileManager::instance()
{
    if (!s_instance) {
        s_instance = new HardwareProfileManager();
    }
    return *s_instance;
}

// ========================================================================
// CONSTRUCTOR / DESTRUCTOR
// ========================================================================

HardwareProfileManager::HardwareProfileManager()
    : SettingsStorage({BC::Key::HardwareProfiles::profiles})
{
    loadProfiles();
}

HardwareProfileManager::HardwareProfileManager(const QString& orgName, const QString& appName)
    : SettingsStorage(orgName, appName, {BC::Key::HardwareProfiles::profiles})
{
    loadProfiles();
}

HardwareProfileManager::~HardwareProfileManager()
{
    if (hasUnsavedChanges()) {
        saveProfiles();
    }
}

// ========================================================================
// PROFILE MANAGEMENT
// ========================================================================

QString HardwareProfileManager::createHardwareProfile(const QString& type, 
                                                     const QString& implementation,
                                                     const QString& requestedLabel,
                                                     CollisionAction collisionAction)
{
    QWriteLocker locker(&d_profilesLock);
    
    // Validate inputs
    if (type.isEmpty() || implementation.isEmpty()) {
        return QString();
    }
    
    // Determine label to use
    QString labelToUse = requestedLabel.trimmed();
    if (labelToUse.isEmpty()) {
        labelToUse = generateDefaultLabelInternal(type);
    }
    
    // Validate label
    if (validateLabelInternal(labelToUse) != Valid) {
        if (requestedLabel.isEmpty()) {
            // Auto-generated label is invalid - this shouldn't happen
            qWarning() << "Generated invalid label for type" << type << ":" << labelToUse;
            return QString();
        } else {
            // User provided invalid label
            return QString();
        }
    }
    
    // Check for collision
    CollisionAction collision = detectCollisionInternal(type, labelToUse, implementation);
    if (collision != NoCollision) {
        switch (collisionAction) {
        case Rename:
            labelToUse = resolveCollisionByRenameInternal(type, labelToUse);
            break;
        case Replace:
            // Will overwrite existing profile
            break;
        case Restore:
        case Cancel:
            return QString();
        case NoCollision:
            break;
        }
    }
    
    // Create the profile
    if (createProfileInternal(type, implementation, labelToUse)) {
        setModified();
        return labelToUse;
    }
    
    return QString();
}

bool HardwareProfileManager::deleteHardwareProfile(const QString& type, const QString& label)
{
    {
        QWriteLocker locker(&d_profilesLock);

        auto typeIt = d_profiles.find(type);
        if (typeIt == d_profiles.end()) {
            return false;
        }

        auto labelIt = typeIt->find(label);
        if (labelIt == typeIt->end()) {
            return false;
        }

        typeIt->erase(labelIt);
        if (typeIt->isEmpty()) {
            d_profiles.erase(typeIt);
        }

        setModified();
    }

    // Purge QSettings for this profile. If the profile has an active hardware object
    // (i.e., it is in the current runtime config), removeHardwareInternal() will call
    // purgeSettings() on the live object when the sync runs — this handles the
    // d_discard flag needed to suppress ~SettingsStorage() re-writing.
    // If there is no live hardware object (inactive profile), purge directly now.
    QString hwKey = BC::Key::hwKey(type, label);
    auto activeHardware = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    if (activeHardware.find(hwKey) == activeHardware.end()) {
        SettingsStorage::purgeGroup({hwKey});
        SettingsStorage::purgeGroupsBySuffix(hwKey);
    }

    return true;
}

bool HardwareProfileManager::activateHardwareProfile(const QString& type, const QString& label)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return false;
    }
    
    if (!labelIt->active) {
        labelIt->active = true;
        updateModificationTime(type, label);
        setModified();
    }
    
    return true;
}

bool HardwareProfileManager::deactivateHardwareProfile(const QString& type, const QString& label)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return false;
    }
    
    if (labelIt->active) {
        labelIt->active = false;
        updateModificationTime(type, label);
        setModified();
    }
    
    return true;
}

bool HardwareProfileManager::profileExists(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    return typeIt->contains(label);
}

bool HardwareProfileManager::isProfileActive(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return false;
    }
    
    return labelIt->active;
}

// ========================================================================
// LABEL MANAGEMENT
// ========================================================================

bool HardwareProfileManager::isLabelAvailable(const QString& type, const QString& label) const
{
    // Empty or invalid labels are never available
    if (validateLabelInternal(label) != Valid) {
        return false;
    }
    
    return !profileExists(type, label);
}

QString HardwareProfileManager::generateDefaultLabel(const QString& type) const
{
    QReadLocker locker(&d_profilesLock);
    return generateDefaultLabelInternal(type);
}

QStringList HardwareProfileManager::getExistingLabels(const QString& type) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return QStringList();
    }
    
    return typeIt->keys();
}

HardwareProfileManager::LabelValidationError HardwareProfileManager::validateLabel(const QString& label) const
{
    return validateLabelInternal(label);
}

bool HardwareProfileManager::isValidLabel(const QString& label) const
{
    return validateLabelInternal(label) == Valid;
}

// ========================================================================
// PROFILE QUERIES
// ========================================================================

QStringList HardwareProfileManager::getActiveProfiles(const QString& type) const
{
    QReadLocker locker(&d_profilesLock);
    
    QStringList activeProfiles;
    auto typeIt = d_profiles.find(type);
    if (typeIt != d_profiles.end()) {
        for (auto it = typeIt->begin(); it != typeIt->end(); ++it) {
            if (it->active) {
                activeProfiles.append(it.key());
            }
        }
    }
    
    return activeProfiles;
}

QStringList HardwareProfileManager::getInactiveProfiles(const QString& type) const
{
    QReadLocker locker(&d_profilesLock);
    
    QStringList inactiveProfiles;
    auto typeIt = d_profiles.find(type);
    if (typeIt != d_profiles.end()) {
        for (auto it = typeIt->begin(); it != typeIt->end(); ++it) {
            if (!it->active) {
                inactiveProfiles.append(it.key());
            }
        }
    }
    
    return inactiveProfiles;
}

QStringList HardwareProfileManager::getAllProfiles(const QString& type) const
{
    return getExistingLabels(type);
}

QString HardwareProfileManager::getImplementation(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return QString();
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return QString();
    }
    
    return labelIt->implementation;
}

QStringList HardwareProfileManager::getConfiguredHardwareTypes() const
{
    QReadLocker locker(&d_profilesLock);
    return d_profiles.keys();
}

// ========================================================================
// PROFILE METADATA
// ========================================================================

QDateTime HardwareProfileManager::getProfileCreationTime(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return QDateTime();
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return QDateTime();
    }
    
    return labelIt->created;
}

QDateTime HardwareProfileManager::getProfileLastModified(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return QDateTime();
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return QDateTime();
    }
    
    return labelIt->modified;
}

bool HardwareProfileManager::setProfileDescription(const QString& type, const QString& label, const QString& description)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return false;
    }
    
    labelIt->description = description;
    updateModificationTime(type, label);
    setModified();
    return true;
}

QString HardwareProfileManager::getProfileDescription(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return QString();
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return QString();
    }
    
    return labelIt->description;
}

// ========================================================================
// COLLISION HANDLING
// ========================================================================

HardwareProfileManager::CollisionAction HardwareProfileManager::detectCollision(const QString& type, const QString& label, 
                                                                               const QString& implementation) const
{
    QReadLocker locker(&d_profilesLock);
    return detectCollisionInternal(type, label, implementation);
}

QString HardwareProfileManager::resolveCollisionByRename(const QString& type, const QString& baseLabel) const
{
    QReadLocker locker(&d_profilesLock);
    return resolveCollisionByRenameInternal(type, baseLabel);
}

// ========================================================================
// BULK OPERATIONS
// ========================================================================

bool HardwareProfileManager::activateAllProfiles(const QString& type)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    bool modified = false;
    for (auto it = typeIt->begin(); it != typeIt->end(); ++it) {
        if (!it->active) {
            it->active = true;
            it->modified = QDateTime::currentDateTime();
            modified = true;
        }
    }
    
    if (modified) {
        setModified();
    }
    
    return true;
}

bool HardwareProfileManager::deactivateAllProfiles(const QString& type)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    bool modified = false;
    for (auto it = typeIt->begin(); it != typeIt->end(); ++it) {
        if (it->active) {
            it->active = false;
            it->modified = QDateTime::currentDateTime();
            modified = true;
        }
    }
    
    if (modified) {
        setModified();
    }
    
    return true;
}

bool HardwareProfileManager::deleteAllProfiles(const QString& type)
{
    QWriteLocker locker(&d_profilesLock);
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return false;
    }
    
    d_profiles.erase(typeIt);
    setModified();
    return true;
}

void HardwareProfileManager::clearAllProfiles()
{
    QWriteLocker locker(&d_profilesLock);
    
    if (!d_profiles.isEmpty()) {
        d_profiles.clear();
        setModified();
    }
}

// ========================================================================
// IMPORT/EXPORT FUNCTIONALITY  
// ========================================================================

QByteArray HardwareProfileManager::exportProfiles() const
{
    QReadLocker locker(&d_profilesLock);
    
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    
    // Write version for future compatibility
    stream << static_cast<quint32>(1);
    
    // Write profile data manually (QDataStream doesn't support nested QHash serialization)
    stream << static_cast<quint32>(d_profiles.size());
    for (auto typeIt = d_profiles.begin(); typeIt != d_profiles.end(); ++typeIt) {
        stream << typeIt.key(); // Hardware type
        stream << static_cast<quint32>(typeIt->size());
        
        for (auto profileIt = typeIt->begin(); profileIt != typeIt->end(); ++profileIt) {
            stream << profileIt.key(); // Profile label
            const ProfileInfo& profile = profileIt.value();
            stream << profile.implementation;
            stream << profile.active;
            stream << profile.created;
            stream << profile.modified;
            stream << profile.description;
        }
    }
    
    return data;
}

QByteArray HardwareProfileManager::exportProfiles(const QString& type) const
{
    QReadLocker locker(&d_profilesLock);
    
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    
    // Write version
    stream << static_cast<quint32>(1);
    
    // Write single type manually
    auto typeIt = d_profiles.find(type);
    if (typeIt != d_profiles.end()) {
        stream << static_cast<quint32>(1); // One type
        stream << type;
        stream << static_cast<quint32>(typeIt->size());
        
        for (auto profileIt = typeIt->begin(); profileIt != typeIt->end(); ++profileIt) {
            stream << profileIt.key(); // Profile label
            const ProfileInfo& profile = profileIt.value();
            stream << profile.implementation;
            stream << profile.active;
            stream << profile.created;
            stream << profile.modified;
            stream << profile.description;
        }
    } else {
        stream << static_cast<quint32>(0); // No types
    }
    
    return data;
}

bool HardwareProfileManager::importProfiles(const QByteArray& data, CollisionAction collisionAction)
{
    if (data.isEmpty()) {
        return false;
    }
    
    QDataStream stream(data);
    
    // Read version
    quint32 version;
    stream >> version;
    
    if (version != 1) {
        qWarning() << "Unsupported profile data version:" << version;
        return false;
    }
    
    // Read profile data manually
    QHash<QString, QHash<QString, ProfileInfo>> importData;
    
    quint32 typeCount;
    stream >> typeCount;
    
    for (quint32 t = 0; t < typeCount; ++t) {
        QString type;
        stream >> type;
        
        quint32 profileCount;
        stream >> profileCount;
        
        QHash<QString, ProfileInfo> typeProfiles;
        for (quint32 p = 0; p < profileCount; ++p) {
            QString label;
            ProfileInfo profile;
            
            stream >> label;
            stream >> profile.implementation;
            stream >> profile.active;
            stream >> profile.created;
            stream >> profile.modified;
            stream >> profile.description;
            
            typeProfiles[label] = profile;
        }
        
        importData[type] = typeProfiles;
    }
    
    QWriteLocker locker(&d_profilesLock);
    
    // Import profiles with collision handling
    for (auto typeIt = importData.begin(); typeIt != importData.end(); ++typeIt) {
        const QString& type = typeIt.key();
        const auto& typeProfiles = typeIt.value();
        
        for (auto profileIt = typeProfiles.begin(); profileIt != typeProfiles.end(); ++profileIt) {
            QString label = profileIt.key();
            const ProfileInfo& profile = profileIt.value();
            
            // Handle collision
            CollisionAction collision = detectCollisionInternal(type, label, profile.implementation);
            if (collision != NoCollision) {
                switch (collisionAction) {
                case Rename:
                    label = resolveCollisionByRenameInternal(type, label);
                    break;
                case Replace:
                    // Will overwrite
                    break;
                case Restore:
                    continue; // Skip this profile
                case Cancel:
                    return false;
                case NoCollision:
                    break;
                }
            }
            
            // Import the profile
            d_profiles[type][label] = profile;
        }
    }
    
    setModified();
    return true;
}

bool HardwareProfileManager::importProfile(const HardwareProfileData& profileData, 
                                          CollisionAction collisionAction)
{
    if (profileData.type.isEmpty() || profileData.label.isEmpty() || profileData.implementation.isEmpty()) {
        return false;
    }
    
    QWriteLocker locker(&d_profilesLock);
    
    QString label = profileData.label;
    
    // Handle collision
    CollisionAction collision = detectCollisionInternal(profileData.type, label, profileData.implementation);
    if (collision != NoCollision) {
        switch (collisionAction) {
        case Rename:
            label = resolveCollisionByRenameInternal(profileData.type, label);
            break;
        case Replace:
            // Will overwrite
            break;
        case Restore:
        case Cancel:
            return false;
        case NoCollision:
            break;
        }
    }
    
    // Create profile info
    ProfileInfo profile;
    profile.implementation = profileData.implementation;
    profile.active = profileData.active;
    profile.created = profileData.created.isValid() ? profileData.created : QDateTime::currentDateTime();
    profile.modified = profileData.modified.isValid() ? profileData.modified : QDateTime::currentDateTime();
    profile.description = profileData.description;
    
    d_profiles[profileData.type][label] = profile;
    setModified();
    return true;
}

HardwareProfileData HardwareProfileManager::getProfileData(const QString& type, const QString& label) const
{
    QReadLocker locker(&d_profilesLock);
    
    HardwareProfileData data;
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return data;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return data;
    }
    
    data.type = type;
    data.label = label;
    data.implementation = labelIt->implementation;
    data.active = labelIt->active;
    data.created = labelIt->created;
    data.modified = labelIt->modified;
    data.description = labelIt->description;
    
    return data;
}

// ========================================================================
// PERSISTENCE MANAGEMENT
// ========================================================================

void HardwareProfileManager::saveProfiles()
{
    QReadLocker locker(&d_profilesLock);
    saveProfilesToSettings();
    
    QMutexLocker modifiedLocker(&d_modifiedFlagLock);
    d_modified = false;
}

void HardwareProfileManager::loadProfiles()
{
    QWriteLocker locker(&d_profilesLock);
    loadProfilesFromSettings();
    
    QMutexLocker modifiedLocker(&d_modifiedFlagLock);
    d_modified = false;
}

bool HardwareProfileManager::hasUnsavedChanges() const
{
    QMutexLocker locker(&d_modifiedFlagLock);
    return d_modified;
}

bool HardwareProfileManager::isSystemProfile(const QString& hwType, const QString& label)
{
    return label == QStringLiteral("virtual") && RuntimeHardwareConfig::isHardwareRequired(hwType);
}

void HardwareProfileManager::ensureSystemProfiles()
{
    // Build the required-type -> virtual-impl map
    QMap<QString, QString> requiredVirtualMap;
    requiredVirtualMap[QString(VirtualFtmwScope::staticMetaObject.className())] =
        QString(VirtualFtmwScope::staticMetaObject.className());
    requiredVirtualMap[QString(FixedClock::staticMetaObject.className())] =
        QString(FixedClock::staticMetaObject.className());

    if (ApplicationConfigManager::instance().isLifEnabled()) {
        requiredVirtualMap[QString(VirtualLifScope::staticMetaObject.className())] =
            QString(VirtualLifScope::staticMetaObject.className());
        requiredVirtualMap[QString(VirtualLifLaser::staticMetaObject.className())] =
            QString(VirtualLifLaser::staticMetaObject.className());
    }

    // The map keys above are the implementation class names, but the hardware type keys
    // are the BASE class names. We need to look them up from the registry.
    // FtmwScope, Clock, LifScope, LifLaser are the base type names.
    // Build a map from hwType -> virtualImpl using RuntimeHardwareConfig::isHardwareRequired
    // and the registry to find the type each virtual class belongs to.
    HardwareRegistry& registry = HardwareRegistry::instance();
    QStringList allTypes = registry.getHardwareTypes();

    for (const QString& hwType : allTypes) {
        if (!RuntimeHardwareConfig::isHardwareRequired(hwType)) {
            continue;
        }

        // Find the virtual implementation for this type
        // The virtual implementations are named Virtual<Type> or Fixed<Type>
        // Check registry for the known virtual impl names
        QStringList impls = registry.getImplementations(hwType);
        QString virtualImpl;

        // Check each known virtual impl to see if it's registered for this type
        for (auto it = requiredVirtualMap.begin(); it != requiredVirtualMap.end(); ++it) {
            if (impls.contains(it.value())) {
                virtualImpl = it.value();
                break;
            }
        }

        if (virtualImpl.isEmpty()) {
            qWarning() << "HardwareProfileManager::ensureSystemProfiles: No virtual implementation found for required type" << hwType;
            continue;
        }

        // Create the "virtual" profile if it doesn't already exist
        if (!getAllProfiles(hwType).contains(QStringLiteral("virtual"))) {
            QString actualLabel = createHardwareProfile(hwType, virtualImpl, QStringLiteral("virtual"), Replace);
            if (actualLabel.isEmpty()) {
                qWarning() << "HardwareProfileManager::ensureSystemProfiles: Failed to create system profile for" << hwType;
            } else {
                qDebug() << "HardwareProfileManager::ensureSystemProfiles: Created system profile" << hwType << "virtual ->" << virtualImpl;
            }
        }
    }
}

// ========================================================================
// INTERNAL HELPER METHODS
// ========================================================================

void HardwareProfileManager::loadProfilesFromSettings()
{
    d_profiles.clear();
    
    // Use SettingsStorage groupKeys() to get all profile groups
    // Expected format: Groups like "FlowController.frontPanel", "FlowController.backup"
    QStringList allGroupKeys = this->groupKeys();
    
    for (const QString& groupKey : allGroupKeys) {
        // Parse group key format: type.label (using dot separator to avoid conflicts)
        QStringList parts = groupKey.split('.');
        if (parts.size() != 2) continue;
        
        const QString& type = parts[0];
        const QString& label = parts[1];
        
        // Ensure profile structure exists
        if (!d_profiles.contains(type)) {
            d_profiles[type] = QHash<QString, ProfileInfo>();
        }
        
        ProfileInfo profile;
        profile.implementation = getGroupValue<QString>(groupKey, BC::Key::HardwareProfiles::implementation, QString());
        profile.active = getGroupValue<bool>(groupKey, BC::Key::HardwareProfiles::active, true);
        
        QString createdStr = getGroupValue<QString>(groupKey, BC::Key::HardwareProfiles::created, QString());
        profile.created = QDateTime::fromString(createdStr, Qt::ISODate);
        if (!profile.created.isValid()) {
            profile.created = QDateTime::currentDateTime();
        }
        
        QString modifiedStr = getGroupValue<QString>(groupKey, BC::Key::HardwareProfiles::modified, QString());
        profile.modified = QDateTime::fromString(modifiedStr, Qt::ISODate);
        if (!profile.modified.isValid()) {
            profile.modified = QDateTime::currentDateTime();
        }
        
        profile.description = getGroupValue<QString>(groupKey, BC::Key::HardwareProfiles::description, QString());
        
        // Only add if we have a valid implementation that exists in the registry
        if (!profile.implementation.isEmpty() && 
            HardwareRegistry::instance().getImplementations(type).contains(profile.implementation)) {
            d_profiles[type][label] = profile;
        }
    }
}

void HardwareProfileManager::saveProfilesToSettings()
{
    // Clear existing group data to avoid stale entries
    QStringList existingGroupKeys = groupKeys();
    for (const QString& groupKey : existingGroupKeys) {
        clearValue(groupKey);
    }
    
    // Write all profile data using SettingsStorage GroupValues
    for (auto typeIt = d_profiles.begin(); typeIt != d_profiles.end(); ++typeIt) {
        const QString& type = typeIt.key();
        const auto& typeProfiles = typeIt.value();
        
        for (auto profileIt = typeProfiles.begin(); profileIt != typeProfiles.end(); ++profileIt) {
            const QString& label = profileIt.key();
            const ProfileInfo& profile = profileIt.value();
            
            // Create group key: type.label (e.g., "FlowController.frontPanel")
            QString groupKey = QString("%1.%2").arg(type, label);
            
            // Set all properties for this profile group
            setGroupValue(groupKey, BC::Key::HardwareProfiles::implementation, profile.implementation, false);
            setGroupValue(groupKey, BC::Key::HardwareProfiles::active, profile.active, false);
            setGroupValue(groupKey, BC::Key::HardwareProfiles::created, profile.created.toString(Qt::ISODate), false);
            setGroupValue(groupKey, BC::Key::HardwareProfiles::modified, profile.modified.toString(Qt::ISODate), false);
            if (!profile.description.isEmpty()) {
                setGroupValue(groupKey, BC::Key::HardwareProfiles::description, profile.description, false);
            }
        }
    }
    
    // Save all changes to persistent storage
    save();
}


void HardwareProfileManager::setModified()
{
    QMutexLocker locker(&d_modifiedFlagLock);
    d_modified = true;
}

HardwareProfileManager::LabelValidationError HardwareProfileManager::validateLabelInternal(const QString& label) const
{
    if (label.isEmpty() || label.trimmed().isEmpty()) {
        return Empty;
    }
    
    if (label.length() > getMaxLabelLength()) {
        return TooLong;
    }
    
    // Check for dots (conflicts with settings key format)
    if (label.contains('.')) {
        return ContainsDots;
    }
    
    // Check for internal spaces
    if (label.contains(' ')) {
        return InvalidCharacters;
    }
    
    // Check if starts with number
    if (label.at(0).isDigit()) {
        return StartsWithNumber;
    }
    
    // Check if starts with underscore
    if (label.startsWith('_')) {
        return StartsWithUnderscore;
    }
    
    // Check for valid characters (alphanumeric, underscore, dash)
    QRegularExpression validChars("^[a-zA-Z][a-zA-Z0-9_-]*$");
    if (!validChars.match(label).hasMatch()) {
        return InvalidCharacters;
    }
    
    return Valid;
}

HardwareProfileManager::CollisionAction HardwareProfileManager::detectCollisionInternal(const QString& type, const QString& label,
                                                                                       const QString& implementation) const
{
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return NoCollision;
    }
    
    auto labelIt = typeIt->find(label);
    if (labelIt == typeIt->end()) {
        return NoCollision;
    }
    
    // Collision exists - could be replace or restore depending on implementation
    if (labelIt->implementation == implementation) {
        return Restore; // Same implementation, might want to restore
    } else {
        return Replace; // Different implementation, might want to replace
    }
}

bool HardwareProfileManager::createProfileInternal(const QString& type, const QString& implementation, 
                                                   const QString& label)
{
    ProfileInfo profile(implementation, true);
    d_profiles[type][label] = profile;
    return true;
}

QString HardwareProfileManager::generateDefaultLabelInternal(const QString& type) const
{
    auto typeIt = d_profiles.find(type);
    QStringList existingLabels;
    if (typeIt != d_profiles.end()) {
        existingLabels = typeIt->keys();
    }
    
    // Try standard default labels
    QStringList candidates = {"Default", "Main", "Primary", "Secondary", "Backup"};
    
    for (const QString& candidate : candidates) {
        if (!existingLabels.contains(candidate)) {
            return candidate;
        }
    }
    
    // Generate numbered label as fallback
    int counter = 1;
    QString candidate;
    do {
        candidate = QString("Device%1").arg(counter);
        counter++;
    } while (existingLabels.contains(candidate) && counter <= 1000);
    
    return candidate;
}

void HardwareProfileManager::updateModificationTime(const QString& type, const QString& label)
{
    auto typeIt = d_profiles.find(type);
    if (typeIt != d_profiles.end()) {
        auto labelIt = typeIt->find(label);
        if (labelIt != typeIt->end()) {
            labelIt->modified = QDateTime::currentDateTime();
        }
    }
}

QString HardwareProfileManager::resolveCollisionByRenameInternal(const QString& type, const QString& baseLabel) const
{
    // Assumes caller already holds the lock
    QString newLabel = baseLabel;
    int counter = 1;
    
    auto typeIt = d_profiles.find(type);
    if (typeIt == d_profiles.end()) {
        return newLabel; // No profiles of this type, no collision
    }
    
    while (typeIt->contains(newLabel)) {
        newLabel = QString("%1_%2").arg(baseLabel).arg(counter);
        counter++;
        
        // Prevent infinite loop
        if (counter > 1000) {
            newLabel = QString("%1_%2").arg(baseLabel).arg(QDateTime::currentMSecsSinceEpoch());
            break;
        }
    }
    
    return newLabel;
}