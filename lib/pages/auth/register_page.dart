import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/form_input.dart';
import '../../widgets/primary_button.dart';
import '../recruiter/recruiter_main_page.dart';
import '../seeker/seeker_shell.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key, this.initialRole = 'job_seeker'});
  final String initialRole;

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final fullNameCtrl = TextEditingController();
  final emailCtrl = TextEditingController();
  final passwordCtrl = TextEditingController();
  final confirmCtrl = TextEditingController();
  final locationCtrl = TextEditingController();
  final headlineCtrl = TextEditingController();
  final skillsCtrl = TextEditingController();
  final yearsCtrl = TextEditingController();
  final companyNameCtrl = TextEditingController();
  final companyLocationCtrl = TextEditingController();
  late String selectedRole;
  String? emailError;
  String? passwordError;
  String? confirmError;

  @override
  void initState() {
    super.initState();
    selectedRole = widget.initialRole;
  }

  @override
  void dispose() {
    fullNameCtrl.dispose();
    emailCtrl.dispose();
    passwordCtrl.dispose();
    confirmCtrl.dispose();
    locationCtrl.dispose();
    headlineCtrl.dispose();
    skillsCtrl.dispose();
    yearsCtrl.dispose();
    companyNameCtrl.dispose();
    companyLocationCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final isRecruiter = selectedRole == 'recruiter';
    return Scaffold(
      appBar: AppBar(title: const Text('Create account')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isRecruiter ? 'Recruiter account' : 'Job seeker account',
                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 16),
              FormInput(
                  label: isRecruiter ? 'Contact name' : 'Full name',
                  controller: fullNameCtrl,
                  hint: 'Alex Doe'),
              const SizedBox(height: 12),
              FormInput(
                  label: 'Email',
                  controller: emailCtrl,
                  hint: 'alex@example.com',
                  keyboardType: TextInputType.emailAddress),
              if (emailError != null) _inlineError(emailError!),
              const SizedBox(height: 12),
              Row(children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      FormInput(
                          label: 'Password',
                          controller: passwordCtrl,
                          obscure: true,
                          hint: '••••••••'),
                      if (passwordError != null) _inlineError(passwordError!),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      FormInput(
                          label: 'Confirm',
                          controller: confirmCtrl,
                          obscure: true,
                          hint: '••••••••'),
                      if (confirmError != null) _inlineError(confirmError!),
                    ],
                  ),
                ),
              ]),
              const SizedBox(height: 12),
              if (isRecruiter) ...[
                FormInput(
                    label: 'Company name',
                    controller: companyNameCtrl,
                    hint: 'TechVision Solutions'),
                const SizedBox(height: 12),
                FormInput(
                    label: 'Company location (Country/City)',
                    controller: companyLocationCtrl,
                    hint: 'Singapore'),
              ] else ...[
                FormInput(
                    label: 'Preferred location (Country/City)',
                    controller: locationCtrl,
                    hint: 'Kuala Lumpur'),
                const SizedBox(height: 12),
                FormInput(
                    label: 'Headline / target role',
                    controller: headlineCtrl,
                    hint: 'Frontend Engineer'),
                const SizedBox(height: 12),
                FormInput(
                    label: 'Key skills',
                    controller: skillsCtrl,
                    hint: 'React, Flutter, APIs'),
                const SizedBox(height: 12),
                FormInput(
                    label: 'Years of experience',
                    controller: yearsCtrl,
                    hint: '4',
                    keyboardType: TextInputType.number),
              ],
              const SizedBox(height: 20),
              PrimaryButton(
                label: 'Create account',
                loading: auth.isLoading,
                onPressed: () async {
                  final email = emailCtrl.text.trim();
                  final password = passwordCtrl.text;
                  final confirm = confirmCtrl.text;
                  setState(() {
                    emailError = _requiredError(email);
                    passwordError = _requiredError(password);
                    confirmError = _requiredError(confirm);
                  });
                  if (password != confirm) {
                    final mismatchMessage =
                        (password.isEmpty || confirm.isEmpty)
                            ? _friendlyMessage('required')
                            : 'Passwords do not match.';
                    setState(() {
                      confirmError ??= mismatchMessage;
                    });
                    _showError(context, mismatchMessage);
                    return;
                  }
                  await auth.register(
                    fullName: fullNameCtrl.text,
                    email: email,
                    password: password,
                    role: selectedRole,
                    preferredLocation:
                        isRecruiter ? null : locationCtrl.text,
                    headline: isRecruiter ? null : headlineCtrl.text,
                    skills: isRecruiter ? null : skillsCtrl.text,
                    experienceYears:
                        isRecruiter ? null : int.tryParse(yearsCtrl.text),
                    companyName:
                        isRecruiter ? companyNameCtrl.text : null,
                    companyLocation:
                        isRecruiter ? companyLocationCtrl.text : null,
                  );
                  if (!mounted) return;
                  if (auth.session != null) {
                    _goToDashboard(context, auth);
                    context.read<JobProvider>().loadJobs();
                  } else if (auth.error != null) {
                    _applyServerError(auth.error!);
                  }
                },
              ),
              const SizedBox(height: 12),
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Back to sign in',
                    style: TextStyle(
                        color: AppTheme.primary, fontWeight: FontWeight.w700)),
              )
            ],
          ),
        ),
      ),
    );
  }

  void _goToDashboard(BuildContext context, AuthProvider auth) {
    final role = auth.session?.profile.role;
    if (role == 'recruiter') {
      Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => const RecruiterMainPage()),
          (_) => false);
    } else {
      Navigator.pushAndRemoveUntil(context,
          MaterialPageRoute(builder: (_) => const SeekerShell()), (_) => false);
    }
  }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, maxLines: 2, overflow: TextOverflow.ellipsis),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  String? _requiredError(String value) {
    if (value.trim().isEmpty) {
      return _friendlyMessage('required');
    }
    return null;
  }

  String _friendlyMessage(String message) {
    final lower = message.toLowerCase();
    if (lower.contains('too short')) {
      return 'Password must be at least 6 characters.';
    }
    if (lower.contains('invalid email')) {
      return 'Please enter a valid email address.';
    }
    if (lower.contains('required')) {
      return 'Please fill in this field.';
    }
    return message;
  }

  void _applyServerError(String message) {
    final friendly = _friendlyMessage(message);
    final lower = message.toLowerCase();
    setState(() {
      if (lower.contains('invalid email') || lower.contains('email')) {
        emailError = friendly;
        return;
      }
      if (lower.contains('too short') || lower.contains('password')) {
        passwordError = friendly;
        return;
      }
      if (lower.contains('confirm')) {
        confirmError = friendly;
        return;
      }
      if (lower.contains('required')) {
        if (emailCtrl.text.trim().isEmpty) {
          emailError = friendly;
        }
        if (passwordCtrl.text.isEmpty) {
          passwordError = friendly;
        }
        if (confirmCtrl.text.isEmpty) {
          confirmError = friendly;
        }
      }
    });
    _showError(context, friendly);
  }

  Widget _inlineError(String message) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, top: 6),
      child: Text(
        message,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(
            color: Colors.red, fontWeight: FontWeight.w600, fontSize: 12),
      ),
    );
  }
}
