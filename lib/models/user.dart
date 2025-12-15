class UserProfile {
  final String id;
  final String email;
  final String fullName;
  final String role;
  final String? preferredLocation;
  final String? headline;
  final String? skills;
  final int? experienceYears;

  const UserProfile({
    required this.id,
    required this.email,
    required this.fullName,
    required this.role,
    this.preferredLocation,
    this.headline,
    this.skills,
    this.experienceYears,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      id: json['id'] ?? json['user_id'] ?? '',
      email: json['email'] ?? '',
      fullName: json['full_name'] ?? '',
      role: json['role'] ?? 'job_seeker',
      preferredLocation: json['preferred_location'],
      headline: json['headline'],
      skills: json['skills'],
      experienceYears: json['experience_years'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'full_name': fullName,
      'role': role,
      'preferred_location': preferredLocation,
      'headline': headline,
      'skills': skills,
      'experience_years': experienceYears,
    };
  }
}

class UserSession {
  final UserProfile profile;
  final String token;

  const UserSession({required this.profile, required this.token});
}
