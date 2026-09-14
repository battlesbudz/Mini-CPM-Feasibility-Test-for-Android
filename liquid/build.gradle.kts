plugins { id("com.android.application") }
android {
    namespace = "com.battlesbudz.liquidtest"
    compileSdk = 35
    ndkVersion = "27.2.12479018"
    defaultConfig {
        applicationId = "com.battlesbudz.liquidtest"
        minSdk = 29
        targetSdk = 35
        versionCode = System.getenv("GITHUB_RUN_NUMBER")?.toIntOrNull() ?: 1
        versionName = "0.1.0-build.$versionCode"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON", "-DCMAKE_BUILD_TYPE=Release", "-DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc"); targets += "liquid_bench" } }
        buildConfigField("String", "SOURCE_COMMIT", "\"${System.getenv("GITHUB_SHA") ?: "local"}\"")
    }
    signingConfigs { getByName("debug") { storeFile = rootProject.file("signing/feasibility-test.jks") } }
    buildTypes { release { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("debug") } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    buildFeatures { buildConfig = true }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
}
dependencies { testImplementation("junit:junit:4.13.2") }
